"""Every absolute data location the code reads or writes. Scripts import this module (after putting this lib folder first
on sys.path) and take their paths from it. The data are not part of the git repository: they live in its git-ignored
data/ folder (DATA), by kind: raw/ (VULCAN packs), caches/, truth/, sealed/ (never list or read), runs/, benches/, ref/,
gen/ (the clean generator's working state).

DATA comes from the environment variable CHEM_DATA (empty = unset; default <this repository>/data). Other overrides:
ARCH_CACHE (data.CACHE, rawtraj.CACHE_DIR; default ARCH_TESTS), HARNESS_NORM (data.NORM; default NORM12), GCM_RAW
(cache_gcm; default RAW_GCM), HARNESS_OUT (where speed.py writes; default OUT).
"""
import json
import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]           # this repository
LIB = HARNESS / "lib"
DATA = Path(os.environ.get("CHEM_DATA") or HARNESS / "data")   # all project data (git-ignored)
RAW_GCM = DATA / "raw/raw_gcm"                           # packed gcm.py raw files (cache_gcm default GCM_RAW)
RAW_CLEAN = DATA / "raw/raw_clean"                       # packed clean-design raw files (build_clean.sh GCM_RAW)

VREF = DATA / "ref/vulcan"                               # VULCAN reference tags: where a tag NAME given to lhbench resolves
SNAPSHOT = DATA / "ref/snapshot"                         # frozen Auto-Chem snapshot (orphaned commit 7d5b036), pinned dependency
SNAP_PROCESSING = SNAPSHOT / "processing"                # preprocessing.py (rawtraj)
NORM12 = SNAPSHOT / "data/processed/normalization.json"  # production normalization (12 species); data.NORM default
E = DATA / "benches/E13_gcm"                             # E13 benches: H_*/P*_ model outputs
GTEST, GTEST_COLD = DATA / "truth/gtest", DATA / "truth/gtest_cold"   # fixed-condition truth sets (G, Gc)
PATHS_V2S, PATHS_V2L, PATHS_V2X = DATA / "truth/paths_v2s", DATA / "truth/paths_v2l", DATA / "truth/paths_v2x"   # v2s, v2l, xlong
PATHS_V3 = DATA / "truth/paths_v3"                       # chain sets: dev (DEV, PD), hold (PH), train_a, train_b
                                                         # (hold2 = PH2 is sealed, in sealed/: never list or read it)
RUNS = DATA / "runs"                                     # training runs (CBg, its stage 1 CBS1, FINAL, LD, LD1, ...); speed.py
OUT = Path(os.environ.get("HARNESS_OUT") or DATA / "benches/out")   # speed.py outputs
T2 = DATA / "truth/t2"                                   # T2 snap-back test: tag/ (VULCAN boxes, jobs.json; the T2 design's source)
GCM_TP = DATA / "ref/gcm_tp.npz"                         # GCM T-P occupancy of the published GCMs (the envelope env_any)
GEN = DATA / "ref/never_rerun"                           # targeted row lists rows_*.npy (never-rerun rule)
CHAINS_CLEAN = DATA / "truth/chains"                     # clean stage-2 training chains (train_a, train_b)
PANELS = DATA / "truth/panels/decision"                  # the decision set's test-set histories
DESIGNS = DATA / "benches/designs"                       # score designs: decision/ (panels S, L, P and readouts),
                                                         # t2, PD, PH, PX, P2l, P2s (bench/score_design.py)

ARCH_TESTS = DATA / "caches/arch_gcm17m_res"             # default ARCH_CACHE (its raw_test.npz is lhbench's test split)
ARCH_CLEAN = DATA / "caches/arch_clean_res"              # the clean training cache (25 entries; build_clean.sh)
HOLDOUTS = {"h2": DATA / "truth/e13_holdout/h2", "h2c": DATA / "truth/e13_holdout/h2c",      # opened fixed-condition sets
            "h3": DATA / "truth/e13_holdout3/h3", "h3c": DATA / "truth/e13_holdout3/h3c",
            "h4": DATA / "sealed/e13_holdout4/h4", "h4c": DATA / "sealed/e13_holdout4/h4c"}   # h4, h4c: sealed (never list or read)
PATHMAP = DATA / "ref/pathmap.json"                      # {old absolute location: its place under DATA}, for records older than DATA


def moved(x):
    """x with every absolute path that names a location from before the data moved under DATA (a str, or a list of
    them, as in a stored run config) replaced by where that file is now (PATHMAP, longest matching old prefix); anything
    else is returned as is."""
    if isinstance(x, list):
        return [moved(v) for v in x]
    if not isinstance(x, str) or not x.startswith("/"):
        return x
    m = json.loads(PATHMAP.read_text())
    old = max((o for o in m if x == o or x.startswith(o + "/")), key=len, default=None)
    return x if old is None else str(DATA / m[old]) + x[len(old):]
