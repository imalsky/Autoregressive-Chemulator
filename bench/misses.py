"""Miss readout on the cache's validation runs: how often, where and by how much a model misses VULCAN (quick tests).

    python bench/misses.py <run dir> [stage file] [--n N] [--confirm] [--split train] [--out FILE]   (env as trained)
    python bench/misses.py --compare A_rows.npz B_rows.npz              (paired: the same rows, two models)

A miss is a scored point with an error above 0.1 dex, a large miss above 1 dex (scorerule, SCORE_RULE phys: per-gas
floors, resolved points = number density >= 1e-2 cm^-3). Rows: N validation runs (one fixed permutation of the validation
split, the same for every model: its first N runs screen; --confirm uses the next N, disjoint, kept for confirmation), each
rolled at its constant T, P from a start age t0 in T0S (0 = the exact start state, row 0; otherwise the
stored trajectory at t0) with one dt in DTS for K = min(1000, max(10, round(1000 / dt))) calls, against the stored
trajectory after every call (the training interpolant RawSplit.logy_at: 8.9 stored rows per decade, log-log between them,
so label error is included). Per row: the worst scored error over its calls and the 12 gases, and the averaged error (the
12-gas mean over the calls, for reference). Reported over all rows, per t0, dt, T band and start kind (the clean design's
design_map, joined on SEED): rows, share with a miss, share with a large miss, p99, p99.9 and max of the worst error, the
mean averaged error; the summed fractional error per row (10^e - 1 over its calls and the 12 gases; mean over rows and per call); and the 20
worst rows. Three sets are reported beside all rows: judged = the main domain (C/O <= CO_MAIN, and perturbed starts
scrambled by at most SIGMA_JUDGED dex per species); co_side = the C/O side check above CO_MAIN; stress = perturbed starts
scrambled harder (the last two reported, not judged).
Model calls as in the benches: fp64 state, TF32 off, 32 Newton iterations with
the LM fallback. Writes <out>.json (default <run dir>/misses_<stage>.json) and the per-row arrays <out>_rows.npz.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import train  # noqa: E402,F401  (sets the entry set of models from the normalization file)
import data as D  # noqa: E402
import models as M  # noqa: E402
import scorerule as SR  # noqa: E402
import chains as C  # noqa: E402

T0S = (0.0, 1.0, 10.0, 100.0, 1000.0)
DTS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0)
BANDS = ((300, 650), (650, 1300), (1300, 2000), (2000, 3001))
MISS, LARGE = 0.1, 1.0
SIGMA_JUDGED = 1.0   # dex: perturbed starts scrambled harder than this form the stress set
CO_MAIN = 1.0        # C/O above this is the side check
SEED = 20261023
DESIGN_MAP = paths.DATA / "gen" / "clean" / "design_map.npy"


def model_of(run, stage, dev):
    run = Path(run)
    man = json.loads((run / "manifest.json").read_text())
    got = hashlib.sha256(D.NORM_PATH.read_bytes()).hexdigest()
    if got != man["norm"]["sha256"]:
        sys.exit(f"{run}: HARNESS_NORM {D.NORM_PATH} is not the normalization this model was trained with")
    blob = torch.load(run / stage, map_location=dev)
    _, _, zlo, zhi = D.norm_consts(dev)
    m = M.build(blob["cfg"]["head"], blob["cfg"]["hidden"], zlo, zhi, newton=32, fallback="lm",
                latent=blob["cfg"].get("fm_latent", 512)).to(dev)
    m.load_state_dict(blob["state_dict"])
    return m.eval()


def summary(r, m):
    """Over the rows m: rows; share of rows with a miss / a large miss; share of calls with a miss / a large miss; p99 /
    p99.9 / max of the rows' worst error; mean of the averaged error."""
    w, a = r["worst"][m], r["avg"][m]
    if w.size == 0:
        return {"rows": 0}
    nc = r["ncall"][m].sum()
    return {"rows": int(w.size), "miss_share": float((w > MISS).mean()), "large_share": float((w > LARGE).mean()),
            "call_miss_share": float(r["nmiss"][m].sum() / nc), "call_large_share": float(r["nlarge"][m].sum() / nc),
            "frac_sum_mean": float(r["frac"][m].mean()), "frac_sum_p50": float(np.median(r["frac"][m])),
            "frac_per_call": float(r["frac"][m].sum() / nc),
            "p99": float(np.percentile(w, 99)), "p999": float(np.percentile(w, 99.9)), "max": float(w.max()),
            "mean_err": float(a.mean())}


@torch.no_grad()
def readout(m, n, dev, confirm=False, split="validation"):
    val = D.RawSplit(split, dev)   # --split train: the same readout on training runs (is the model data-limited?)
    mu, sd, _, _ = D.norm_consts(dev)
    cols = [D.SHORT.index(s) for s in C.REPORTED]
    floor = torch.tensor([SR.PHYS[s] for s in C.REPORTED], dtype=torch.float64, device=dev)
    gen = torch.Generator(device=dev).manual_seed(SEED)
    perm = torch.randperm(val.n, device=dev, generator=gen)
    idx_all = perm[n:2 * n] if confirm else perm[:n]
    lnd_all = torch.from_numpy(np.log10(val.P / (C.KB * val.T))).to(dev)
    dm = np.load(DESIGN_MAP)
    kind_of = dict(zip(dm["SEED"].tolist(), dm["kind"].tolist()))
    zc_of = dict(zip(dm["SEED"].tolist(), zip(dm["Z"].tolist(), dm["C_O"].tolist())))
    sigma_of = dict(zip(dm["SEED"].tolist(), dm["sigma"].tolist()))
    rec = {k: [] for k in ("run", "t0", "dt", "worst", "avg", "gas", "call", "ncall", "nmiss", "nlarge", "frac")}
    for t0 in T0S:
        for dt in DTS:
            K = min(1000, max(10, round(1000.0 / dt)))
            ok = val.t[val.end[idx_all]] >= t0 + K * dt
            idx = idx_all[ok]
            B = idx.numel()
            z = val.z_of(val.logy[idx, 0]) if t0 == 0 else \
                val.z_of(val.logy_at(idx, torch.full((B, 1), t0, dtype=torch.float64, device=dev)))[:, 0]
            g, lnd = val.g[idx], lnd_all[idx][:, None]
            dtn = D.dt_norm_of(torch.full((B,), dt, dtype=torch.float64, device=dev)).float()
            worst = torch.zeros(B, dtype=torch.float64, device=dev)
            wgas = torch.zeros(B, dtype=torch.long, device=dev)
            wcall = torch.zeros(B, dtype=torch.long, device=dev)
            avg = torch.zeros(B, dtype=torch.float64, device=dev)
            nmiss = torch.zeros(B, dtype=torch.long, device=dev)
            nlarge = torch.zeros(B, dtype=torch.long, device=dev)
            frac = torch.zeros(B, dtype=torch.float64, device=dev)
            for k in range(1, K + 1):
                z = z + m.delta(z.float(), dtn, g).double()
                lt = val.logy_at(idx, torch.full((B, 1), t0 + k * dt, dtype=torch.float64, device=dev))[:, 0, cols]
                lp = (z * sd + mu)[:, cols]
                res = ((lt + lnd) >= -2.0) | ((lp + lnd) >= -2.0)
                cnt = res & ((lt >= floor) | (lp >= floor))
                e = torch.where(cnt, (torch.maximum(lp, floor) - torch.maximum(lt, floor)).abs(), torch.zeros_like(lp))
                e = torch.where(res & torch.logical_not(torch.isfinite(lp)), torch.full_like(e, math.inf), e)
                v, gi = e.max(-1)
                up = v > worst
                worst, wgas, wcall = torch.where(up, v, worst), torch.where(up, gi, wgas), torch.where(up, k, wcall)
                avg += e.mean(-1) / K
                nmiss += v > MISS
                nlarge += v > LARGE
                frac += (torch.pow(10.0, e) - 1.0).sum(-1)   # fractional error in the worse direction, summed over gases
            rec["run"].append(idx.cpu().numpy())
            rec["t0"].append(np.full(B, t0))
            rec["dt"].append(np.full(B, dt))
            rec["ncall"].append(np.full(B, K))
            for key, v in (("worst", worst), ("avg", avg), ("gas", wgas), ("call", wcall), ("nmiss", nmiss), ("nlarge", nlarge),
                           ("frac", frac)):
                rec[key].append(v.cpu().numpy())
    r = {k: np.concatenate(v) for k, v in rec.items()}
    seeds = np.load(D.CACHE / f"raw_{split}.npz")["seed"]   # the generator's SEED of each run
    r["T"] = val.T[r["run"]]
    r["Pbar"] = val.P[r["run"]] / 1e6
    r["kind"] = np.array([kind_of.get(int(s), "?") for s in seeds[r["run"]]])
    zc = np.array([zc_of.get(int(s), (np.nan, np.nan)) for s in seeds[r["run"]]])
    r["Z"], r["C_O"] = zc[:, 0], zc[:, 1]
    r["sigma"] = np.array([sigma_of.get(int(s), np.nan) for s in seeds[r["run"]]])
    return r


def stress_mask(r):
    """Rows of perturbed starts scrambled harder than SIGMA_JUDGED (the stress set)."""
    return (r["kind"] == "pert") & (r["sigma"] > SIGMA_JUDGED)


def set_masks(r):
    """judged / co_side / stress row masks (module docstring)."""
    st = stress_mask(r)
    co = np.logical_not(st) & (r["C_O"] > CO_MAIN)
    return {"judged": np.logical_not(st | co), "co_side": co, "stress": st}


def report(r):
    w = r["worst"]
    out = {"all": summary(r, np.ones(w.size, bool)),
           "by_set": {k: summary(r, m) for k, m in set_masks(r).items()},
           "by_t0": {f"{t:g}": summary(r, r["t0"] == t) for t in T0S},
           "by_dt": {f"{d:g}": summary(r, r["dt"] == d) for d in DTS},
           "by_T": {f"{lo}-{hi}": summary(r, (r["T"] >= lo) & (r["T"] < hi)) for lo, hi in BANDS},
           "by_kind": {k: summary(r, r["kind"] == k) for k in np.unique(r["kind"])},
           "by_t0_dt": {f"{t:g}|{d:g}": summary(r, (r["t0"] == t) & (r["dt"] == d)) for t in T0S for d in DTS}}
    top = np.argsort(-w)[:20]
    out["worst_rows"] = [{"t0": float(r["t0"][i]), "dt": float(r["dt"][i]), "call": int(r["call"][i]),
                          "kind": str(r["kind"][i]), "T": float(r["T"][i]), "Pbar": float(r["Pbar"][i]),
                          "Z": float(r["Z"][i]), "C_O": float(r["C_O"][i]), "gas": C.REPORTED[int(r["gas"][i])],
                          "err": float(w[i])} for i in top]
    return out


def show(out):
    def line(lab, s):
        return (f"{lab:>14}: rows {s['rows']:6d}  miss {100 * s['miss_share']:6.2f}%  large {100 * s['large_share']:6.3f}%  "
                f"calls {100 * s['call_miss_share']:6.3f}% / {100 * s['call_large_share']:6.4f}%  "
                f"frac sum/row {s.get('frac_sum_mean', float('nan')):9.3g} per call {s.get('frac_per_call', float('nan')):8.3g}  "
                f"p99 {s['p99']:.3f}  p99.9 {s['p999']:.2f}  max {s['max']:.2f}  avg {s['mean_err']:.5f}")
    print(line("all", out["all"]))
    for grp in ("by_set", "by_t0", "by_dt", "by_T", "by_kind"):
        for k, s in out[grp].items():
            if s["rows"]:
                print(line(f"{grp[3:]} {k}", s))


def compare(fa, fb):
    """Paired: rows with a miss / large miss under A and not under B and the reverse, and both shares."""
    A, B = np.load(fa), np.load(fb)
    for k in ("run", "t0", "dt"):
        assert np.array_equal(A[k], B[k]), f"{k}: the two files do not hold the same rows"
    sets = {"all rows": np.ones(A["worst"].size, bool)}
    if "sigma" in A.files:
        sets.update(set_masks(A))
    for name, m in sets.items():
        wa, wb, ra = A["worst"][m], B["worst"][m], A["run"][m]
        print(f"{name}:")
        for th in (MISS, LARGE):
            ma, mb = wa > th, wb > th
            print(f"  > {th:g} dex: A {100 * ma.mean():.3f}%  B {100 * mb.mean():.3f}%  only A {int((ma & np.logical_not(mb)).sum())}"
                  f"  only B {int((mb & np.logical_not(ma)).sum())}  of {wa.size} rows; runs with one: A "
                  f"{np.unique(ra[ma]).size}  B {np.unique(ra[mb]).size}")
        print(f"  worst error: A p99 {np.percentile(wa, 99):.3f} max {wa.max():.2f} | B p99 {np.percentile(wb, 99):.3f} max {wb.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--split", default="validation", choices=("validation", "train"))
    ap.add_argument("--out")
    ap.add_argument("--cont", action="store_true")   # the model's third fallback (models.py cont)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    run = Path(a.run)
    stage = a.stage or ("stage2.pt" if (run / "stage2.pt").exists() else "stage1.pt")
    out = Path(a.out) if a.out else run / (f"misses_{Path(stage).stem}{'_confirm' if a.confirm else ''}"
                                           f"{'_train' if a.split == 'train' else ''}{'_cont' if a.cont else ''}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    m = model_of(run, stage, dev)
    m.cont = a.cont
    r = readout(m, a.n, dev, a.confirm, a.split)
    rep = {"model": str(run), "stage": stage, "n_runs": a.n, "confirm": a.confirm, "split": a.split, "cont": a.cont, "seed": SEED, "t0s": T0S, "dts": DTS, **report(r)}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    np.savez(str(out) + "_rows.npz", **{k: v for k, v in r.items()})
    show(rep)


if __name__ == "__main__":
    main()
