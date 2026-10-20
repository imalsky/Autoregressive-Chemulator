"""58-day paths (paths_v2x: 16 paths x 10,000 calls at dt 500 s, T and P change every call, 50 forcing periods) on the
overall-error cells of scoreboard.path_board, overall and by elapsed window, to see whether error grows with time.
    python xlong_board.py <run> [<run> ...]   (a run given as PX1:<run> reads that run's stage-1 bench PX1_<run>)"""
import json
import os
import sys

import numpy as np

import scoreboard as S

WIN = [(50, 1000), (1000, 3000), (3000, 10001)]   # calls; 1000 calls = 5.8 days
res = {}
for run in sys.argv[1:]:
    f = S.E / (f"PX1_{run[4:]}" if run.startswith("PX1:") else f"PX_{run}") / "P_paths.npz"
    d = np.load(f)
    st, calls = S.A.cut12(d["states"][:, 2]), d["calls"]
    vt, rs = [], []
    for k, p in enumerate(d["path"]):
        q = np.load(S.paths.PATHS_V2X / f"{int(p)}.npz")
        cols = [list(q["species"]).index(s) for s in S.A.D.SHORT]
        x = np.log10(np.maximum(q["x"][calls][:, cols], S.A.D.EPS))
        n = q["Pbar"][calls - 1] * 1e6 / (S.A.KB * q["T"][calls - 1])
        vt.append(x)
        rs.append((x + np.log10(n)[:, None] >= -2.0) | (st[:, k] + np.log10(n)[:, None] >= -2.0))
    vt, rs = np.stack(vt, 1), np.stack(rs, 1)
    e = S.SR.err(vt, st, rs, S.SP)   # [C, paths, 12]
    r = res[run] = {"checkpoints": int(calls.size), "paths": int(e.shape[1])}
    for lo, hi in WIN:
        m = (calls >= lo) & (calls < hi)
        q = e[m][np.isfinite(e[m])]
        r[f"calls {lo}-{hi}"] = {"n_ckpt": int(m.sum()), "p50": float(np.percentile(q, 50)), "p90": float(np.percentile(q, 90)),
                                 "p99": float(np.percentile(q, 99)), "max": float(q.max()),
                                 "paths_gt0.5": int((np.nan_to_num(e[m], nan=0.0).max(axis=(0, 2)) > 0.5).sum())}
    pm = np.nan_to_num(e[calls >= 50], nan=0.0).max(axis=(0, 2))
    k = np.unravel_index(np.nanargmax(np.where(calls[:, None, None] >= 50, e, np.nan)), e.shape)
    Tm = np.array([np.load(S.paths.PATHS_V2X / f"{int(p)}.npz")["T"].mean() for p in d["path"]])
    r["per_path_max"] = {int(p): [round(float(v), 2), round(float(t))] for p, v, t in zip(d["path"], pm, Tm)}
    r["worst"] = {"dex": float(e[k]), "call": int(calls[k[0]]), "path": int(d["path"][k[1]]), "species": S.SP[k[2]],
                  "pred": float(st[k]), "truth": float(vt[k])}
    print(f"== {run} ({calls.size} checkpoints, {e.shape[1]} paths) worst {r['worst']}")
    for lo, hi in WIN:
        c = r[f"calls {lo}-{hi}"]
        print(f"  calls {lo:5d}-{hi:5d} ({c['n_ckpt']} ckpt) p50 {c['p50']:.3f} p90 {c['p90']:.3f} p99 {c['p99']:.3f} max {c['max']:.2f} paths>0.5 {c['paths_gt0.5']}/{e.shape[1]}")
    print("  per-path max [dex, mean T]:", r["per_path_max"])
(S.OUT / os.environ.get("XLB_OUT", "xlong_board.json")).write_text(json.dumps(res, indent=1))
