"""Changing-condition path sets (cells as the former xlong board), for any paths folder benched with
lhbench arm P (2026-10-09 conservation phase: the paths_v3 DEV set, whose recorded mixing the bench applies).
    python path_board.py <paths dir> <run> [<run> ...] [--prefix PD] [--windows 50-1000,1000-2500,2500-] [--vs <run>]
A run's bench is E/<prefix>_<run>/P_paths.npz; the prefix defaults from the folder name (dev -> PD, paths_v2x -> PX,
paths_v2l -> P2l, paths_v2s -> P2s). Truth: the folder's own path files (x[c] after c calls). Points: the 12
reported species (from 12-species, reservoir-set or 53-species states), truth or prediction >= 1e-2 cm^-3 and the
mixing-ratio floor of lib/scorerule.py (per-gas floors, SCORE_RULE=phys, the default since 2026-10-19). Per window of calls (the last one may be open-ended: to the path end): p50, p90, p99,
max and paths with any point > 0.5 dex; per-path maxima over calls >= the first window's start, with the path's mean T;
the worst point. --vs <run>: each run minus <run> on the late (last) window, paired over paths, 1000 bootstrap resamples
of the paths (seed 0), 95% percentile intervals: (a) per-path p50 and p99, mean over paths of the differences; (b) the
board's pooled p50 and p99 (points of all resampled paths). Negative = the run is better. Writes
paths.OUT/path_board_<prefix>.json (PB_OUT sets the name).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import scorerule as SR  # noqa: E402  (2026-10-11: SCORE_RULE selects the scoring rule)

N12 = json.loads(paths.NORM12.read_text())
REP = [s.replace("_evolution", "") for s in N12["species_variables"]]
SPRES = [s.replace("_evolution", "") for s in json.loads(paths.RES_NORM.read_text())["species_variables"]]
EPS, KB, FLOOR = float(N12["epsilon"]), 1.380649e-16, -12.0
PREFIX = {"dev": "PD", "paths_v2x": "PX", "paths_v2l": "P2l", "paths_v2s": "P2s"}
NB, SEED = 1000, 0

ap = argparse.ArgumentParser()
ap.add_argument("pdir")
ap.add_argument("runs", nargs="+")
ap.add_argument("--prefix", default=None)
ap.add_argument("--windows", default="50-1000,1000-2500,2500-")
ap.add_argument("--vs", default=None)
a = ap.parse_args()
pdir = Path(a.pdir)
prefix = a.prefix or PREFIX[pdir.name]
win = [(int(lo), int(hi) if hi else None) for lo, hi in (w.split("-") for w in a.windows.split(","))]


def errors(run):
    """|dlog10| [C, paths, 12] on the scored points (NaN elsewhere), the recorded calls, path ids and mean T."""
    d = np.load(paths.E / f"{prefix}_{run}" / "P_paths.npz")
    if Path(str(d["p_dir"])).resolve() != pdir.resolve():
        sys.exit(f"{run} was benched on {d['p_dir']}, not on {pdir}: refusing to score against other truth")
    calls, st = d["calls"], d["states"][:, 2]
    vt, ok = [], []
    for k, p in enumerate(d["path"]):
        q = np.load(pdir / f"{int(p)}.npz")
        sp = list(q["species"])
        names = {12: REP, len(SPRES): SPRES, len(sp): sp}[st.shape[-1]]
        s = st[:, k][:, [names.index(x) for x in REP]]
        x = np.log10(np.maximum(q["x"][calls][:, [sp.index(x) for x in REP]], EPS))
        ln = np.log10(q["Pbar"][calls - 1] * 1e6 / (KB * q["T"][calls - 1]))[:, None]
        vt.append(SR.err(x, s, ((x + ln) >= -2.0) | ((s + ln) >= -2.0), REP))
    e = np.stack(vt, 1)
    return e, calls, d["path"], d["T"].mean(1)


def wmask(calls, lo, hi):
    return (calls >= lo) & (calls < (hi if hi is not None else calls.max() + 1))


def wname(lo, hi, calls):
    return f"calls {lo}-{hi if hi is not None else int(calls.max())}"


def wq(v, w, q):
    """Weighted percentile (inverted CDF) of sorted values v with weights w."""
    c = np.cumsum(w)
    return v[min(np.searchsorted(c, q / 100.0 * c[-1]), v.size - 1)]


res, E = {}, {}
for run in dict.fromkeys(a.runs + ([a.vs] if a.vs else [])):
    e, calls, pids, Tm = E[run] = errors(run)
    r = res[run] = {"bench": f"{prefix}_{run}", "checkpoints": int(calls.size), "paths": int(e.shape[1])}
    for lo, hi in win:
        m = wmask(calls, lo, hi)
        q = e[m][~np.isnan(e[m])]   # inf (a failed, non-finite prediction) stays in
        r[wname(lo, hi, calls)] = {"n_ckpt": int(m.sum()), "p50": float(np.percentile(q, 50)), "p90": float(np.percentile(q, 90)),
                                   "p99": float(np.percentile(q, 99)), "max": float(q.max()),
                                   "paths_gt0.5": int((np.nan_to_num(e[m], nan=0.0).max(axis=(0, 2)) > 0.5).sum()),
                                   "nonfinite_paths": int(np.isinf(e[m]).any(axis=(0, 2)).sum())}
    first = calls >= win[0][0]
    pm = np.nan_to_num(e[first], nan=0.0, posinf=np.inf).max(axis=(0, 2))
    r["per_path_max"] = {int(p): [round(float(v), 2), round(float(t))] for p, v, t in zip(pids, pm, Tm)}
    k = np.unravel_index(np.nanargmax(np.where(first[:, None, None], e, np.nan)), e.shape)
    r["worst"] = {"dex": float(e[k]), "call": int(calls[k[0]]), "path": int(pids[k[1]]), "species": REP[k[2]]}
    print(f"== {run} ({prefix}, {calls.size} checkpoints, {e.shape[1]} paths) worst {r['worst']}")
    for lo, hi in win:
        c = r[wname(lo, hi, calls)]
        print(f"  {wname(lo, hi, calls):17s} ({c['n_ckpt']} ckpt) p50 {c['p50']:.3f} p90 {c['p90']:.3f} p99 {c['p99']:.3f} "
              f"max {c['max']:.2f} paths>0.5 {c['paths_gt0.5']}/{e.shape[1]}"
              + (f" | NON-FINITE predictions in {c['nonfinite_paths']} paths" if c["nonfinite_paths"] else ""))
    print("  per-path max [dex, mean T]:", r["per_path_max"])

if a.vs:
    ev, cv, pv, _ = E[a.vs]
    lo, hi = win[-1]
    rng = np.random.default_rng(SEED)
    res["vs"] = {"ref": a.vs, "window": wname(lo, hi, cv), "resamples": NB, "seed": SEED}
    for run in a.runs:
        if run == a.vs:
            continue
        e, calls, pids, _ = E[run]
        assert np.array_equal(pids, pv) and np.array_equal(calls, cv), f"{run} and {a.vs} cover different paths or calls"
        R = pids.size
        idx = rng.integers(0, R, (NB, R))
        cnt = np.stack([np.bincount(i, minlength=R) for i in idx])   # [NB, R] times each path is drawn
        late = (e[wmask(calls, lo, hi)], ev[wmask(cv, lo, hi)])
        sv = []                                                      # pooled: sorted values and the path of each, per side
        for x in late:
            f = ~np.isnan(x)
            o = np.argsort(x[f], kind="stable")
            sv.append((x[f][o], np.broadcast_to(np.arange(R)[None, :, None], x.shape)[f][o]))
        out = {}
        for q in (50, 99):
            pp = [np.array([np.percentile(x[:, r][~np.isnan(x[:, r])], q) if (~np.isnan(x[:, r])).any() else np.nan
                            for r in range(R)]) for x in late]
            dp = pp[0] - pp[1]                                       # per path, run - ref
            bm = np.nanmean(dp[idx], 1)
            bp = np.array([wq(sv[0][0], c[sv[0][1]], q) - wq(sv[1][0], c[sv[1][1]], q) for c in cnt])
            pool = float(np.percentile(sv[0][0], q) - np.percentile(sv[1][0], q))
            out[f"p{q}"] = {"per_path_mean_diff": float(np.nanmean(dp)), "per_path_ci95": [float(v) for v in np.percentile(bm, [2.5, 97.5])],
                            "paths_better": int((dp < 0).sum()), "paths_worse": int((dp > 0).sum()),
                            "pooled_diff": pool, "pooled_ci95": [float(v) for v in np.percentile(bp, [2.5, 97.5])],
                            "per_path_diff": {int(p): round(float(v), 4) for p, v in zip(pids, dp)}}
        res["vs"][run] = out
        print(f"-- {run} minus {a.vs}, {res['vs']['window']}, paired over {R} paths (bootstrap {NB}, seed {SEED}):")
        for q in (50, 99):
            o = out[f"p{q}"]
            print(f"  p{q}: per-path mean {o['per_path_mean_diff']:+.4f} [{o['per_path_ci95'][0]:+.4f}, {o['per_path_ci95'][1]:+.4f}]"
                  f" (better {o['paths_better']}, worse {o['paths_worse']}); pooled {o['pooled_diff']:+.4f} "
                  f"[{o['pooled_ci95'][0]:+.4f}, {o['pooled_ci95'][1]:+.4f}]")
(paths.OUT / os.environ.get("PB_OUT", f"path_board_{prefix}.json")).write_text(json.dumps(res, indent=1))
