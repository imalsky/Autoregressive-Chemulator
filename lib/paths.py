"""Every absolute data location the code reads or writes. Scripts import this module (after putting this lib folder first
on sys.path) and take their paths from it. The data are not part of the git repository: they live in its git-ignored
data/ folder (DATA), by kind: raw/ (VULCAN packs), caches/, truth/, sealed/ (never list or read), runs/, benches/, ref/,
gen/ (the clean generator's working state).

DATA comes from the environment variable CHEM_DATA (empty = unset; default <this repository>/data).
"""
import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]           # this repository
LIB = HARNESS / "lib"
DATA = Path(os.environ.get("CHEM_DATA") or HARNESS / "data")   # all project data (git-ignored)
RAW_CLEAN = DATA / "raw/raw_clean"                       # packed clean-design VULCAN runs (build_kin.py)
GCM_TP = DATA / "ref/gcm_tp.npz"                         # GCM T-P occupancy of the published GCMs (the envelope env_any)
GEN = DATA / "ref/never_rerun"                           # used row lists and SEEDs (new data must avoid them)
CHAINS_CLEAN = DATA / "truth/chains"                     # clean training chains (train_a, train_b)
PANELS = DATA / "truth/panels/decision"                  # the decision set's test-set histories
DESIGNS = DATA / "benches/designs"                       # score designs: decision/ (panels S, L, P and readouts;
                                                         # bench/score_design.py)
