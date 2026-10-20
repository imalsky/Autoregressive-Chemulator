"""Every absolute data location the harness reads or writes (2026-10-09: the code moved from the sweep's _build/lib and the
sprint folder into code/e13-harness; data, runs, caches and test sets stayed where they were). Scripts import this module
(after putting this lib folder first on sys.path) and take their paths from it. Note: experiments' E13_gcm folder holds an
unrelated paths.py (the v1 path generator); lib must come first on sys.path for 'import paths' to find this one.

Environment overrides (unchanged from before the move): ARCH_CACHE (data.CACHE, rawtraj.CACHE_DIR), HARNESS_NORM
(data.NORM), GCM_RAW (cache_gcm). New: HARNESS_OUT, where the boards, scorers and speed.py write (default PHASE/_build/bench).
"""
import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]           # this repository
LIB = HARNESS / "lib"
BENCH_MODES = HARNESS / "run" / "BENCH_MODES"            # lhbench's default --modes (P3), was <sweep _build>/BENCH_MODES

PROJECT = Path("/home/imalsky/Chemulator_Project")
SWEEP = PROJECT / "experiments/2026-10-03_repo-sweep/_build"   # 'B': runs, VULCAN references, snapshot, E13 benches
RUNS = SWEEP / "runs"                                    # training runs (<run>/config.json, stage1.pt, stage2.pt, ...)
VREF = SWEEP / "vulcan"                                  # VULCAN reference tags (ref_a, ref_b, ref_c, ref_p); lhbench.VREF
SNAPSHOT = SWEEP / "snapshot"                            # frozen Auto-Chem snapshot (orphaned commit 7d5b036), pinned dependency
SNAP_SRC = SNAPSHOT / "src"                              # model.py (MLP, FlowMapMLP, ResidualMLP), trainer.py (HybridLoss)
SNAP_PROCESSING = SNAPSHOT / "processing"                # preprocessing.py (rawtraj)
NORM12 = SNAPSHOT / "data/processed/normalization.json"  # production normalization (12 species); data.NORM default
SHARDS = SNAPSHOT / "data/processed"                     # production validation shards (data.load_val_shards)
E = SWEEP / "g2/E13_gcm"                                 # E13 bench folder: H_*/P*_/G_stress_* benches, truth sets, rows
GTEST, GTEST_COLD = E / "gtest", E / "gtest_cold"        # fixed-condition truth sets (G, Gc)
PATHS_V2S, PATHS_V2L, PATHS_V2X = E / "paths_v2s", E / "paths_v2l", E / "paths_v2x"   # changing-condition paths (v2s, v2l, xlong)
SOBOL_DIR = SWEEP / "g2/E12_sobol"                       # E12 (cache_sobol.py's old home)

SPRINT = PROJECT / "experiments/2026-10-08_e13-sprint/_build"   # 'X': sprint queues (q/), gen_shift data, weights
GEN_SHIFT = SPRINT / "gen_shift"                         # gen_shift.py sets (<set>/manifest.npz, boxes_<rtol>/), m17_weights.npy
PHASE = PROJECT / "experiments/2026-10-09_e13-conservation"    # current phase; figures go to its top level
PHASE_BUILD = PHASE / "_build"
OUT = Path(os.environ.get("HARNESS_OUT") or PHASE_BUILD / "bench")  # board / scorer / speed outputs (were beside each script)
FIGS = PHASE                                             # paper-style figures (were written to the sprint folder's top level)

SCRATCH = Path("/scratch/imalsky")                       # harness caches (arch_*), pseudo-truth tags (e13_ctruth_*)
ARCH_TESTS = SCRATCH / "arch_tests"                      # default ARCH_CACHE (production-data cache; lhbench's test split)
RES_NORM = SCRATCH / "arch_gcm16_res/norm.json"          # reservoir-set species list (analyze_e13.cut12)

DATAGEN = PROJECT / "code/training-data-generator"       # gcm.py, generate.py (VULCAN driver; imported, not modified)
RAW_GCM = PROJECT / "code/Auto-Chem/data/raw_gcm"        # packed gcm.py raw files (cache_gcm default GCM_RAW)
RAW_SOBOL = PROJECT / "code/Auto-Chem/data/raw_sobol"    # packed Sobol raw files (cache_sobol)
SCIPLOT = Path("/home/imalsky/.claude/skills/scientific-plotting")   # figure style helper (sciplot)
