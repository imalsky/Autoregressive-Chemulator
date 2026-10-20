"""E13: is VULCAN's rtol = 0.05 (the training data's and the test truth's setting) accurate enough for a 0.1 dex target?
The first 128 jobs of the GCM-like test set G (a balanced 2^7 prefix of its Sobol design; a stride-8 subsample is
not: its first run, kept in gtol_midT_stride8, covered only 977-1647 K) are rerun with rtol 0.005 (10x tighter, everything else equal) and
compared with G's truth: per job, the max over checkpoints and tracked species above 1e-10 of |log10 y(0.05) - log10
y(0.005)|, at times 1e-2 to 8.64e7 s, by T band.

    python tolerance.py jobs       gtol/jobs.json
    python tolerance.py score      gtol/score.json and a printed summary
Runs: E8_floor/drive.py gtol/jobs.json gtol 0-127 <cpus> 3600
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
E = paths.E
sys.path.insert(0, str(paths.DATAGEN))
import generate as G  # noqa: E402

OUT, SRC = E / "gtol", E / "gtest"
IDX = list(range(128))
BANDS = ((300, 800), (800, 1300), (1300, 1800), (1800, 2300), (2300, 3001))


def jobs():
    OUT.mkdir(exist_ok=True)
    J = json.loads((SRC / "jobs.json").read_text())
    out = [{**J[i], "cfg": {**J[i]["cfg"], "rtol": "0.005"}, "gtest_index": i} for i in IDX]
    (OUT / "jobs.json").write_text(json.dumps(out))
    print(f"{len(out)} jobs")


def logy(f, times):
    r = np.load(f)
    sp = list(r["species"])
    return np.log10(np.maximum(np.stack([np.interp(times, r["t"], r["x"][:, sp.index(s)]) for s in G.SPECIES_OUT], -1), 1e-30))


def score():
    J = json.loads((OUT / "jobs.json").read_text())
    times = np.logspace(-2, np.log10(8.64e7), 41)
    res = []
    for k, j in enumerate(J):
        if not (OUT / f"{k}.npz").exists():
            continue
        a, b = logy(SRC / f"{j['gtest_index']}.npz", times), logy(OUT / f"{k}.npz", times)
        m = np.maximum(a, b) >= -10.0
        res.append({"i": j["gtest_index"], "T": j["T"], "max_dex": float(np.where(m, np.abs(a - b), 0.0).max()),
                    "late_max_dex": float(np.where(m & (times >= 1e3)[:, None], np.abs(a - b), 0.0).max())})
    (OUT / "score.json").write_text(json.dumps(res))
    T = np.array([r["T"] for r in res])
    for key in ("max_dex", "late_max_dex"):
        v = np.array([r[key] for r in res])
        print(f"{key}: jobs {v.size}, p50 {np.median(v):.4f} p90 {np.percentile(v, 90):.4f} max {v.max():.3f}; > 0.1 dex "
              f"{(v > 0.1).sum()}; by band p90: " + ", ".join(f"{lo}-{hi} {np.percentile(v[(T >= lo) & (T < hi)], 90):.4f}"
                                                         for lo, hi in BANDS if ((T >= lo) & (T < hi)).any()))


if __name__ == "__main__":
    jobs() if sys.argv[1] == "jobs" else score()
