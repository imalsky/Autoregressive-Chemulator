"""Rows of fixed-condition benches with a late (>= 1e3 s) point > 2 dex or any point > 5 dex, per set and label (fixes
phase 2026-10-12): job, start, T, P, late max, first-call max, worst species. Pairs labels side by side on the union of
flagged rows.   SCORE_RULE=phys python rows_gt2.py <set> <label> [<label> ...]      (HARNESS_NORM unset)"""
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/
import scoreboard as S  # noqa: E402

HSETS = {"G": ("g500", "H_gtest_dt500.npz"), "Gc": ("gc500", "H_gtest_cold_dt500.npz"), "H2": ("h2", "H_h2_dt500.npz"),
         "H2c": ("h2c", "H_h2c_dt500.npz"), "H3": ("h3", "H_h3_dt500.npz"), "H3c": ("h3c", "H_h3c_dt500.npz")}
k, labels = sys.argv[1], sys.argv[2:]
tag, fn = HSETS[k]
cols = []
for L in labels:
    f = S.E / f"H_{tag}_{L}" / fn
    B, d = S.bench(f), np.load(f)
    e, e1 = np.nan_to_num(B["e"], nan=0.0), np.nan_to_num(B["e1"], nan=0.0)
    cols.append((d, e.max(axis=(1, 2)), e1.max(axis=(1, 2)), e.max(axis=1).argmax(axis=1)))
d0 = cols[0][0]
for d, *_ in cols[1:]:
    assert np.array_equal(d["job"], d0["job"]) and np.array_equal(d["t0"], d0["t0"])
flag = np.zeros(d0["job"].size, bool)
for _, lm, fm, _ in cols:
    flag |= (lm > 2) | (np.maximum(lm, fm) > 5)
SP = list(S.SP) if hasattr(S, "SP") else None
print(f"{k}: rows flagged (late > 2 or any > 5) in any of {labels}: {int(flag.sum())}; columns late / first (worst species)")
for i in np.flatnonzero(flag):
    s = "  ".join(f"{lm[i]:5.2f} / {fm[i]:5.2f} ({SP[w[i]] if SP else w[i]})" for _, lm, fm, w in cols)
    print(f"  job {int(d0['job'][i]):>10d} t0 {d0['t0'][i]:7.0e} {d0['T'][i]:5.0f} K {d0['P'][i] / 1e6:8.2e} bar   {s}")
