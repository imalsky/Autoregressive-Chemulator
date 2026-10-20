"""Accumulation vs state: restart a model from VULCAN's 12 tracked species at call k of each 58-day path (paths_v2x) and
run 1 and 100 calls with the path's own T, P (dt 500 s), scoring as xlong_board (resolved, 1e-12 floor). If the error of
a fresh restart is the same early and late, the growth seen in the full rollout is accumulation; if it is larger late,
the late states themselves are harder (or the 41 untracked species matter).
    python xlong_restart.py <run> [<checkpoint>, default stage2.pt]   (GPU; writes paths.OUT/xlong_restart_<run>.json)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import data as D  # noqa: E402
import lhbench as LB  # noqa: E402

RUN = sys.argv[1]
CK = sys.argv[2] if len(sys.argv) > 2 else "stage2.pt"
LAB = RUN if CK == "stage2.pt" else f"{RUN}_{CK.split('.')[0]}"
KB, FLOOR, NC = 1.380649e-16, -12.0, 100
dev = torch.device("cuda")
model, _ = LB.load_model(str(paths.RUNS / RUN), CK, dev)
mu, sd, _, _ = D.norm_consts(dev)
Ps = [np.load(paths.PATHS_V2X / f"{p}.npz") for p in range(16)]
K = np.arange(50, 9900, 50)                                       # restart calls
ly = np.log10(np.maximum(np.stack([D.take(p["x"], list(p["species"])) for p in Ps]), D.EPS))   # [16, 10001, 12]
T = np.stack([p["T"] for p in Ps]); Pb = np.stack([p["Pbar"] for p in Ps])
pi, ki = np.repeat(np.arange(16), K.size), np.tile(K, 16)           # rows
z = (torch.as_tensor(ly[pi, ki], device=dev) - mu) / sd
dtn = D.dt_norm_of(torch.full((pi.size,), 500.0, dtype=torch.float64, device=dev)).float()
err = np.full((NC, pi.size, 12), np.nan)
with torch.no_grad():
    for c in range(NC):                                            # call c uses T, P of path call k + c
        g = torch.from_numpy(D.norm_globals(Pb[pi, ki + c] * 1e6, T[pi, ki + c])).to(dev)
        z = z + model.delta(z.float(), dtn, g).double()
        pr = (z * sd + mu).cpu().numpy()
        tr = ly[pi, ki + c + 1]
        ln = np.log10(Pb[pi, ki + c] * 1e6 / (KB * T[pi, ki + c]))[:, None]
        m = (((tr + ln) >= -2) | ((pr + ln) >= -2)) & ((tr >= FLOOR) | (pr >= FLOOR))
        err[c] = np.where(m, np.abs(pr - tr), np.nan)
res = {}
for lab, lo, hi in (("restart 50-1000", 50, 1000), ("restart 1000-3000", 1000, 3000), ("restart 3000-9900", 3000, 9900)):
    rm = (ki >= lo) & (ki < hi) & (T[pi, 0] > 0)
    r = res[lab] = {}
    for c in (1, 10, 100):
        q = err[c - 1][rm]; q = q[np.isfinite(q)]
        r[f"after {c} calls"] = {"p50": float(np.percentile(q, 50)), "p90": float(np.percentile(q, 90)), "p99": float(np.percentile(q, 99)), "max": float(q.max())}
    print(f"{LAB} {lab:18s} " + " | ".join(f"{c}: p50 {v['p50']:.3f} p90 {v['p90']:.3f} p99 {v['p99']:.2f} max {v['max']:.2f}" for c, v in r.items()))
(paths.OUT / f"xlong_restart_{LAB}.json").write_text(json.dumps(res, indent=1))
