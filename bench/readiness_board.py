"""Readiness tests (R-full) on the overall-error cells of scoreboard.py, per T band, beside the run's G dt500 (the
comparator is cut to the same elapsed window for dt 10 s, which covers 1e7 s). 2000 days: truth past its last saved
time is held at its final state (score.vtruth uses np.interp), so that row reads as drift from the 1000-day state.
    python readiness_board.py <run>"""
import json
import sys

import numpy as np

import scoreboard as S

RUN = sys.argv[1]
E = S.E
TESTS = [("G dt500", f"H_g500_{RUN}/H_gtest_dt500.npz"), ("G dt100", f"H_g100_{RUN}/H_gtest_dt100.npz"),
         ("startup", f"R_startup_{RUN}/H_gtest_dt500_startup.npz"), ("rand10_500", f"R_rand_{RUN}/H_gtest_dt125.25_rand10_500.npz"),
         ("dt10 to 1e7 s", f"R_dt10all_{RUN}/H_gtest_dt10.npz"), ("2000 days", f"R_d2000_{RUN}/H_gtest_dt500.npz")]


def cut(B, f, tmax):
    d = np.load(f)
    el = d["tcum"].T if "tcum" in d.files else np.broadcast_to(d["calls"][None, :] * float(d["dt"]), (d["t0"].size, d["calls"].size))
    late = np.median(el, 0) >= 1e3
    keep = np.median(el, 0)[late] <= tmax * 1.0001
    return dict(B, e=B["e"][:, keep])


res = {}
for name, rel in TESTS + [("G dt500 to 1e7 s", f"H_g500_{RUN}/H_gtest_dt500.npz")]:
    f = E / rel
    if not f.exists():
        print(f"{name}: missing {f}")
        continue
    B = S.bench(f)
    if name.endswith("to 1e7 s"):
        B = cut(B, f, 1e7)
    r = res[name] = {f"{lo}-{hi}": S.cells(B, (B["T"] >= lo) & (B["T"] < hi)) for lo, hi in zip(S.EDGES[:-1], S.EDGES[1:])}
    k = np.unravel_index(np.nanargmax(B["e"]), B["e"].shape)
    r["worst"] = {"dex": float(B["e"][k]), "T": float(B["T"][k[0]]), "P_bar": float(B["P"][k[0]]), "species": S.SP[k[2]]}
    print(f"== {name}  (late checkpoints {B['e'].shape[1]}) worst {r['worst']}")
    for b in [f"{lo}-{hi}" for lo, hi in zip(S.EDGES[:-1], S.EDGES[1:])]:
        c = r[b]
        print(f"  {b:9s} p90 {c['p90']:.3f} | worst sp {c['worst_sp']} {c['worst_sp_p90']:.2f} | p99 {c['p99']:.3f} | max {c['max']:.2f}"
              f" | rows>0.5 {c['rows_gt0.5']}/{c['rows']}")
(S.OUT / f"readiness_{RUN}.json").write_text(json.dumps(res, indent=1))
