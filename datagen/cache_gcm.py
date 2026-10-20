"""E13: harness cache from the packed gcm.py raw files (blocks 0 to K-1), through rawtraj._work unchanged
(RAWTRAJ_FLOOR=1: floored, no drop_below rejection; production's split by file and group name; the 12 tracked
species). GCM_KINDS (comma list of eq, mix, pert, broad; default all) keeps only groups of those start types (the
type is the design row's, from the SEED). Groups that hit VULCAN's step cap (end_case 3: truncated, then held
constant) are left out. GCM_SPECIES=all caches every network species (gcm.species() order, He included) instead
of the 12. Each block file is cut into n_per chunks.

    RAWTRAJ_FLOOR=1 ARCH_CACHE=<dir> [GCM_KINDS=eq,mix] python cache_gcm.py chunk <K> <n_per> <i>
    RAWTRAJ_FLOOR=1 ARCH_CACHE=<dir> python cache_gcm.py merge <K> <n_per>
"""
import os
import pickle
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(paths.DATAGEN))
import cache_sobol as CS  # noqa: E402  (merge, and rawtraj as CS.R)
import gcm  # noqa: E402

R = CS.R
CS.RAW = Path(os.environ.get("GCM_RAW", str(paths.RAW_GCM)))  # override: tests
KINDS = set(os.environ.get("GCM_KINDS", "eq,mix,pert,broad").split(","))
FILES = os.environ.get("GCM_FILES")  # sprint 2026-10-08: comma list of K raw basenames under GCM_RAW (else run_gcm_bNN)
if os.environ.get("GCM_SPECIES") == "all":
    import dataclasses
    _cfg0 = R._cfg
    R._cfg = lambda: dataclasses.replace(_cfg0(), species_variables=[s + "_evolution" for s in gcm.species()])


def kind(g):
    """Start type: the group's 'start' attr (sprint 2026-10-08: rows outside the design, e.g. arm M, carry one; for the
    design rows it equals the design's kind), else the design row's (from the SEED)."""
    if "start" in g.attrs:
        return str(g.attrs["start"])
    return gcm.params(int(g.attrs["SEED"]) - gcm.SEED_BASE)["kind"]


def jobs(K, n_per):
    out = []
    files = FILES.split(",") if FILES else [f"run_gcm_b{b:02d}-result.h5" for b in range(K)]
    assert len(files) == K, (files, K)
    for fn in files:
        fp = CS.RAW / fn
        with h5py.File(fp, "r") as f:
            names = [n for n in f.keys() if int(f[n].attrs["end_case"]) != 3 and kind(f[n]) in KINDS]
        step = -(-len(names) // n_per)
        out += [(str(fp), names[k:k + step]) for k in range(0, len(names), step)]
    assert len(out) == K * n_per
    return out


def chunk(K, n_per, i):
    res = R._work(jobs(K, n_per)[i])
    R.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(R.CACHE_DIR / f"part_{i}.pkl", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"chunk {i}: kinds {sorted(KINDS)}, rejects {res[2]}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    chunk(int(a[1]), int(a[2]), int(a[3])) if a[0] == "chunk" else CS.merge(int(a[1]), int(a[2]))
