"""E13: the GCM-like test set G (frozen decision set for the 2026-10-06 targets). 1024 jobs from an independent
scrambled Sobol design (seed 20261007) mapped exactly as code/training-data-generator/gcm.py maps its rows, except
that only the physical start types are used (column 5 scaled into [0, 0.7): 5/7 eq, 2/7 mix). Starts are built by
gcm.start (FastChem equilibria at the source states; work files in this folder). Truth to 1e10 s via vulcan_ref.py
box (E8_floor/drive.py), all 53 species, the same format as utest.

    python make_gtest.py            (writes gtest/jobs.json; refuses to overwrite)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
E = paths.E
sys.path.insert(0, str(paths.DATAGEN))
import generate as G  # noqa: E402
import gcm  # noqa: E402

SEED, N, BASE = 20261007, 1024, 4_250_000_000  # IDs above gcm.py's 4.2e9 + 2^20, below 2^32
out = E / "gtest"
out.mkdir(exist_ok=True)
assert not (out / "jobs.json").exists()
u = qmc.Sobol(d=gcm.D, scramble=True, seed=SEED).random_base2(m=10)
u[:, 5] *= 0.7
np.save(out / "design.npy", u)
gcm.DESIGN, gcm.QMC_SEED, gcm.W = out / "design.npy", SEED, out  # gcm reads these module globals at call time
assert BASE + N < 2 ** 32 and BASE > gcm.SEED_BASE + 2 ** gcm.M_MAX
sp = gcm.species()
jobs = []
for i in range(N):
    p = gcm.params(i)
    x0 = gcm.start(i, p, sp)
    jobs.append({"T": p["T"], "P": p["P"], "seed": BASE + i, "lo": -25.0, "mix": x0,
                 "name": G.group_name(p["T"], p["P"], BASE + i), "cfg": {"trun_min": "1e10", "runtime": "1e10"},
                 "file": "gtest", "Z": p["Z"], "C_O": p["C_O"], "f_NO": p["f_NO"], "start": p["kind"],
                 "sources_T_Pbar": p["src"], "mix_weights": p["w"]})
(out / "jobs.json").write_text(json.dumps(jobs, indent=1))
k = [j["start"] for j in jobs]
print(f"{N} jobs: eq {k.count('eq')}, mix {k.count('mix')}; T {min(j['T'] for j in jobs):.0f}-{max(j['T'] for j in jobs):.0f} K")
