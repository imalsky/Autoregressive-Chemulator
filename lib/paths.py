"""Every absolute data location the code reads or writes. Scripts import this module (after putting this lib folder first
on sys.path) and take their paths from it. Note: the E13_gcm bench folder holds an unrelated paths.py (the v1 path
generator); lib must come first on sys.path for 'import paths' to find this one.

Two roots come from the environment (empty = unset), with the daw defaults:
  CHEM_PROJECT   the project folder: experiment data, runs, truth sets, the frozen snapshot   (/home/imalsky/Chemulator_Project)
  CHEM_SCRATCH   the fast disk: training caches, development and holdout truth tags            (/scratch/imalsky)
Other overrides: ARCH_CACHE (data.CACHE, rawtraj.CACHE_DIR; default ARCH_TESTS), HARNESS_NORM (data.NORM; default NORM12),
GCM_RAW (cache_gcm; default RAW_GCM), HARNESS_OUT (where the boards, scorers and speed.py write; default OUT).
"""
import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]           # this repository
LIB = HARNESS / "lib"
RAW_GCM = HARNESS / "data/raw_gcm"                       # packed gcm.py raw files (cache_gcm default GCM_RAW; gitignored)

PROJECT = Path(os.environ.get("CHEM_PROJECT") or "/home/imalsky/Chemulator_Project")
SCRATCH = Path(os.environ.get("CHEM_SCRATCH") or "/scratch/imalsky")

SWEEP = PROJECT / "experiments/2026-10-03_repo-sweep/_build"   # VULCAN references, snapshot, E13 benches and truth sets
VREF = SWEEP / "vulcan"                                  # VULCAN reference tags (ref_a: read by every bench); lhbench.VREF
SNAPSHOT = SWEEP / "snapshot"                            # frozen Auto-Chem snapshot (orphaned commit 7d5b036), pinned dependency
SNAP_SRC = SNAPSHOT / "src"                              # model.py (MLP, FlowMapMLP, ResidualMLP), trainer.py (HybridLoss)
SNAP_PROCESSING = SNAPSHOT / "processing"                # preprocessing.py (rawtraj)
NORM12 = SNAPSHOT / "data/processed/normalization.json"  # production normalization (12 species); data.NORM default
SHARDS = SNAPSHOT / "data/processed"                     # production validation shards (data.load_val_shards)
E = SWEEP / "g2/E13_gcm"                                 # E13 bench folder: H_*/P*_ benches, truth sets, rows
GTEST, GTEST_COLD = E / "gtest", E / "gtest_cold"        # fixed-condition truth sets (G, Gc)
PATHS_V2S, PATHS_V2L, PATHS_V2X = E / "paths_v2s", E / "paths_v2l", E / "paths_v2x"   # changing-condition paths (v2s, v2l, xlong)

PATHS_V3 = PROJECT / "experiments/2026-10-09_e13-conservation/_build/paths_v3"   # chain sets: dev (DEV, PD), hold (PH),
                                                         # train_a, train_b; hold2 = PH2 is sealed (never list or read it)
SCALE = PROJECT / "experiments/2026-10-13_e13-scale/_build"     # scale phase: the current runs and decision data
RUNS = SCALE / "runs"                                    # training runs (CBg, its stage 1 CBS1, FINAL, LD, LD1); speed.py
OUT = Path(os.environ.get("HARNESS_OUT") or SCALE / "bench")   # board / scorer / judge / speed / 1000-day outputs
T2 = SCALE / "t2"                                        # T2 snap-back test: jobs_sel.json, tag/, times_t2.npy, score_t2*.json
T5_DESIGN = SCALE / "t5/design"                          # score_mask's per-row caches rows_<label>.npz (SCORE_MASK_CACHE)
FIG4 = SCALE / "fig4"                                    # the six mini-chem paper cases: tag/, mc_c<case>_*.npz, model benches
DTSWEEP = SCALE / "dtsweep"                              # step-size sweep: rows_main.json, sw_<run>/, score.{json,txt}, score_rows.npz
GCM_TP = SCALE / "research/gcm/gcm_tp.npz"               # GCM T-P occupancy of the published GCMs (the envelope env_any)
CRITIC_ROWS = SCALE / "research/critic/rows_e13s_FINw1536_s2.npz"   # FINAL's per-row summaries (T2 job selection)
GEN = SCALE / "gen"                                      # targeted row lists rows_*.npy (never-rerun rule)

ARCH_TESTS = SCRATCH / "arch_gcm17m_res"                 # default ARCH_CACHE (its raw_test.npz is lhbench's test split)
RES_NORM = SCRATCH / "arch_gcm16_res/norm.json"          # reservoir-set (25-entry) species list (analyze_e13.cut12, path_board)
DEVALL6 = SCRATCH / "e13_scaledev/devall6"               # combined dev tag G + Gc + H2 + H2c + H3 + H3c (dev benches, sweep)
HOLDOUTS = {"h2": SCRATCH / "e13_holdout/h2", "h2c": SCRATCH / "e13_holdout/h2c",      # opened fixed-condition sets
            "h3": SCRATCH / "e13_holdout3/h3", "h3c": SCRATCH / "e13_holdout3/h3c",
            "h4": SCRATCH / "e13_holdout4/h4", "h4c": SCRATCH / "e13_holdout4/h4c"}   # h4, h4c: sealed (never list or read)
