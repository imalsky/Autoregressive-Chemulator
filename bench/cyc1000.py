"""Model-only 1000-day changing-condition run. No truth past 58 days, so it checks boundedness, not accuracy. Each of the 16 xlong paths (paths_v2x: VULCAN start, per-call T, P, dt,
10,000 calls = 58 days) is repeated until 1000 days of model time (~17 cycles, ~172k calls), as lhbench's P3 mode: fp64
normalized state, z += delta(z fp32, dt, T/P) fp64, a call is rejected (state kept, counted) if non-finite or outside the
log10 envelope [-40, 0.05]. Recorded at every cycle end (same forcing phase): the 12 reported species' log10 mixing ratios,
the sum of the 12, and for the whole entry set the element ratios C/H, O/H, N/H (12-species states: over the 12; reservoir
sets: over all entries). Reported: per path, the change of the reported species between successive cycle ends (max over
species, dex), the largest |d log10 ratio| vs the start, sum12 max, rejected calls, the per-row projection counters
(Newton failed / kept input) and whether the state is finite. Closed heads run with fallback lm (the LM rescue where
Newton fails). The 12 reported gases are picked from the entry set by name (chains.REPORTED).
    HARNESS_NORM=<the run's norm.json> python cyc1000.py <run dir> <out json> [<closed iterations>]   (GPU)"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import chains as C  # noqa: E402
import data as D  # noqa: E402
import lhbench as LB  # noqa: E402
import models as M  # noqa: E402
import paths  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
newton = int(sys.argv[3]) if len(sys.argv) > 3 else None
dev = torch.device("cuda")
model, _ = LB.load_model(run, "stage2.pt", dev, newton)
model.eval()
heads = [m_ for m_ in model.modules() if isinstance(m_, M.RelaxClosedRHead)]
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
mu, sd, _, _ = D.norm_consts(dev)
X = str(paths.PATHS_V2X)
Ps = [np.load(f"{X}/{i}.npz") for i in range(16)]
n = min(len(p["T"]) for p in Ps)
z = (torch.as_tensor(np.log10(np.maximum(np.stack([D.take(p["x"][0], list(p["species"])) for p in Ps]), D.EPS)),
                     dtype=torch.float64, device=dev) - mu) / sd
g = torch.from_numpy(np.stack([D.norm_globals(p["Pbar"][:n] * 1e6, p["T"][:n]) for p in Ps], 1)).to(dev)  # [n, R, 2]
dts = np.stack([np.asarray(p["dt"], np.float64)[:n] for p in Ps], 1)                                     # [n, R]
dtn = D.dt_norm_of(torch.as_tensor(dts, device=dev)).float()
ncyc = int(np.ceil(1000 * 86400.0 / dts.sum(0).min()))
zlo, zhi = (LB.ENV_LO - mu) / sd, (LB.ENV_HI - mu) / sd


def atoms(s):
    a = {}
    for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", s):
        a[el] = a.get(el, 0) + (int(k) if k else 1)
    return a


ent = list(D.SHORT)
res = {"C_res": "C", "N_res": "N", "O_res": "O", "H_res": "H"}
A = np.array([[atoms(s).get(e, 0) if s not in res and s != "M_res" else float(res.get(s) == e) for e in "CHON"]
              for s in ent])                                         # [S, 4] atoms per entry (He entry: 0)
rep = [ent.index(s) for s in C.REPORTED]
assert len(rep) == 12, rep


def snapshot(z):
    ly = (z * sd + mu).cpu().numpy()
    y = 10.0 ** ly
    el = y @ A                                                       # [R, 4] C H O N
    return ly[:, rep], y[:, rep].sum(1), np.log10(el[:, [0, 2, 3]] / el[:, [1]])   # C/H, O/H, N/H


ly0, s0, r0 = snapshot(z)
ends, sums, rats = [ly0], [s0], [r0]
rej = torch.zeros(len(Ps), dtype=torch.int64, device=dev)
for h in heads:
    h.fallback, h.rowcount = "lm", torch.zeros(len(Ps), 2, dtype=torch.int64, device=dev)
t = time.time()
with torch.no_grad():
    for c in range(ncyc):
        for k in range(n):
            new = z + model.delta(z.float(), dtn[k], g[k]).double()
            ok = torch.isfinite(new).all(-1) & (new >= zlo).all(-1) & (new <= zhi).all(-1)
            rej += (~ok).long()
            z = torch.where(ok.unsqueeze(-1), new, z)
        ly, s, r = snapshot(z)
        ends.append(ly); sums.append(s); rats.append(r)
        print(f"cycle {c + 1}/{ncyc} ({(c + 1) * dts.sum(0).mean() / 86400:.0f} d, {time.time() - t:.0f} s): max cycle-end change "
              f"{np.abs(ends[-1] - ends[-2]).max():.3f} dex, sum12 max {s.max():.4f}, rejected {int(rej.sum())}", flush=True)
E, Sm, Rt = np.stack(ends), np.stack(sums), np.stack(rats)           # [ncyc + 1, R, ...]
dc = np.abs(np.diff(E, axis=0)).max(-1)                              # [ncyc, R] per cycle, max over species
o = {"run": run, "closed_iters": newton, "cycles": ncyc, "calls": ncyc * n, "days_min": float(ncyc * dts.sum(0).min() / 86400),
     "rejected": rej.cpu().tolist(), "finite": bool(np.isfinite(E).all()), "fallback": "lm",
     "projfail": sum(h.rowcount for h in heads).cpu().tolist() if heads else None,
     "cycle_change_first": dc[0].round(3).tolist(), "cycle_change_last": dc[-1].round(3).tolist(),
     "cycle_change_last5_max": dc[-5:].max(0).round(3).tolist(),
     "ratio_drift_max": np.abs(Rt - Rt[0]).max(axis=(0, 2)).round(4).tolist(), "sum12_max": Sm.max(0).round(4).tolist(),
     "end_minus_cycle1": np.abs(E[-1] - E[1]).max(-1).round(3).tolist(), "seconds": time.time() - t}
json.dump(o, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in o.items() if k not in ("cycle_change_first",)}))
