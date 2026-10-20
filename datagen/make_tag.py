"""Sprint 2026-10-08 (plan item 4): a G-like truth set (copy of E13_gcm/make_gtest.py with a seed, a size, an ID base,
a row filter and the solver tolerance as arguments). Independent scrambled Sobol design (seed SEED, 2^m rows), mapped as
gcm.py maps its rows with the start type limited to eq/mix (column 5 scaled into [0, 0.7), as G); the first N rows that
pass the filter become jobs (job seed = BASE + design row; design_row recorded). rtol is written into every job cfg
when given (vulcan_ref.box reads it; omitted = generate.py's 0.05, as G). Starts by gcm.start (FastChem; work files
in <out>/fc). Truth by E8_floor/drive.py, as G.

    python make_tag.py <name> <out_dir> <seed> <N> <base> <m> <filter: all|cold|corner> [rtol]
      cold   = T < 1300 K (Gc);  corner = (Z > 30 and C/O > 0.9) or Z > 100 (H2c)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DATAGEN))
import generate as G  # noqa: E402
import gcm  # noqa: E402

name, out, SEED, N, BASE, m, filt = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
rtol = sys.argv[8] if len(sys.argv) > 8 else None
assert BASE > gcm.SEED_BASE + 2 ** gcm.M_MAX and BASE + 2 ** m < 2 ** 32 and filt in ("all", "cold", "corner")
for o, n in {4_250_000_000: 1024, 4_251_000_000: 4096, 4_252_000_000: 1024, 4_253_000_000: 2048,
             4_254_000_000: 1024, 4_256_000_000: 2048, 4_260_000_000: 2 ** 20}.items():
    assert o == BASE or BASE + 2 ** m <= o or BASE >= o + n   # G, Gc, H2, H2c, H3, H3c, gen_shift ID ranges stay apart
out.mkdir(parents=True, exist_ok=True)
assert not (out / "jobs.json").exists(), out / "jobs.json"
u = qmc.Sobol(d=gcm.D, scramble=True, seed=SEED).random_base2(m=m)
u[:, 5] *= 0.7
np.save(out / "design.npy", u)
gcm.DESIGN, gcm.QMC_SEED, gcm.W = out / "design.npy", SEED, out  # gcm reads these module globals at call time


def keep(p):
    if filt == "cold":
        return p["T"] < 1300.0
    if filt == "corner":
        return (p["Z"] > 30.0 and p["C_O"] > 0.9) or p["Z"] > 100.0
    return True


sp = gcm.species()
jobs, npass = [], 0
for i in range(2 ** m):
    p = gcm.params(i)
    if not keep(p):
        continue
    npass += 1
    if len(jobs) == N:
        continue
    x0 = gcm.start(i, p, sp)
    cfg = {"trun_min": "1e10", "runtime": "1e10", **({"rtol": rtol} if rtol else {})}
    jobs.append({"T": p["T"], "P": p["P"], "seed": BASE + i, "lo": -25.0, "mix": x0,
                 "name": G.group_name(p["T"], p["P"], BASE + i), "cfg": cfg, "file": name, "Z": p["Z"],
                 "C_O": p["C_O"], "f_NO": p["f_NO"], "start": p["kind"], "sources_T_Pbar": p["src"],
                 "mix_weights": p["w"], "design_row": i})
assert len(jobs) == N, f"only {len(jobs)} of {N} rows pass the filter in 2^{m}"
txt = json.dumps(jobs, indent=1)
(out / "jobs.json").write_text(txt)
k = [j["start"] for j in jobs]
T = np.array([j["T"] for j in jobs])
print(f"{name}: {N} jobs (filter {filt}: {npass} of {2 ** m} design rows pass), eq {k.count('eq')}, mix {k.count('mix')}; "
      f"T {T.min():.0f}-{T.max():.0f} K; rtol {rtol or 'default 0.05'}; last design row {jobs[-1]['design_row']}; "
      f"jobs.json sha256 {hashlib.sha256(txt.encode()).hexdigest()}")
