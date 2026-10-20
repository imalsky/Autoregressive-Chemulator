"""Small-step test from evolved states (a GCM's chemistry starts mostly near equilibrium): the six
development sets (devall6 tag) from VULCAN's own states at 1e3 s and 1e6 s, constant dt 0.1 s and 1 s, 10,000 calls (1,000 s /
10,000 s), P* numerics without startup substeps (run/bench_dev.sh sets SD0.1 / SD1; outputs E13_gcm/H_sd_dt<dt>_<label>). Per row (job x
start): the max error over every checkpoint and the 12 reported gases (per-gas rule, resolved points, NaN = 0). Counted
inside env_any +-100 K / +-0.1 dex (select_jobs.env_margin, as rules_t5.py) and outside.
    SCORE_RULE=phys HARNESS_NORM= python sd_score.py <label> [<label> ...]   -> per label and dt: rows, rows > 0.5, > 2, p99"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/
import scoreboard as S  # noqa: E402
import select_jobs as SJ  # noqa: E402

assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM")
INSIDE = SJ.env_margin()[0]


def score(label, dt):
    B = S.bench(S.E / f"H_sd_dt{dt}_{label}" / f"H_devall6_dt{dt}.npz")
    nonfinite = int(np.logical_not(np.isfinite(B["st"])).any(axis=(1, 2)).sum())   # crashes must not score 0
    e = np.nan_to_num(np.concatenate([B["e1"], B["e"]], 1), nan=0.0).max(axis=(1, 2))   # [rows]
    ins = np.asarray(INSIDE(B["T"], B["P"]), bool)
    out = {"nonfinite_rows": nonfinite}
    for nm, m in (("inside", ins), ("outside", ~ins)):
        out[nm] = {"rows": int(m.sum()), "gt05": int((e[m] > 0.5).sum()), "gt2": int((e[m] > 2.0).sum()),
                   "p99": float(np.percentile(e[m], 99)) if m.any() else float("nan")}
    return out


if __name__ == "__main__":
    for lab in sys.argv[1:]:
        for dt in ("0.1", "1"):
            o = score(lab, dt)
            print(f"{lab:22s} dt {dt:>3s} s | non-finite rows {o['nonfinite_rows']} | " + " | ".join(
                f"{k}: rows {v['rows']} > 0.5 {v['gt05']} > 2 {v['gt2']} p99 {v['p99']:.3f}" for k, v in o.items() if k != "nonfinite_rows"))
