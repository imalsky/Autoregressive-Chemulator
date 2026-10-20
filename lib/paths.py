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

SCALE = PROJECT / "experiments/2026-10-13_e13-scale/_build"     # scale phase: the current runs and decision data
RUNS = SCALE / "runs"                                    # training runs (CBg, its stage 1 CBS1, FINAL, LD, LD1); speed.py
OUT = Path(os.environ.get("HARNESS_OUT") or SCALE / "bench")   # board / scorer / speed outputs

ARCH_TESTS = SCRATCH / "arch_gcm17m_res"                 # default ARCH_CACHE (its raw_test.npz is lhbench's test split)
RES_NORM = SCRATCH / "arch_gcm16_res/norm.json"          # reservoir-set (25-entry) species list (analyze_e13.cut12, path_board)
