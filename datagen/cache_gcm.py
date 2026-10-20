"""Training cache from the generator's packed raw files (blocks 0 to K-1), through rawtraj._work unchanged
(RAWTRAJ_FLOOR=1: floored, no drop_below rejection; the 12-species code's split by file and group name; the 12 tracked
species). GCM_KINDS (comma list of eq, mix, pert, broad, kick; default eq, mix, pert, broad) keeps only groups of those
start types (the group's 'start' attr, which every packed group carries). Groups that hit VULCAN's step cap (end_case 3:
truncated, then held constant) are left out. GCM_SPECIES=all caches every network species (the packs' 'species' attr,
the generator's order, He included) instead of the 12. Each block file is cut into n_per chunks. Reads the packed files
only (no generator import).

    RAWTRAJ_FLOOR=1 ARCH_CACHE=<dir> [GCM_KINDS=eq,mix] python cache_gcm.py chunk <K> <n_per> <i>
    RAWTRAJ_FLOOR=1 ARCH_CACHE=<dir> python cache_gcm.py merge <K> <n_per>
"""
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import rawtraj as R  # noqa: E402

assert R.FLOOR
RAW = Path(os.environ.get("GCM_RAW", str(paths.RAW_GCM)))  # override: tests
KINDS = set(os.environ.get("GCM_KINDS", "eq,mix,pert,broad").split(","))
FILES = os.environ.get("GCM_FILES")  # comma list of K raw basenames under GCM_RAW (else run_gcm_bNN)


def pack_species(f):
    """The network species of an open packed file, in stored order (its 'species' attr)."""
    return [s.decode() for s in f.attrs["species"]]


SPECIES_ALL = None
if os.environ.get("GCM_SPECIES") == "all":
    import dataclasses
    with h5py.File(RAW / (FILES.split(",")[0] if FILES else "run_gcm_b00-result.h5"), "r") as _f:
        SPECIES_ALL = pack_species(_f)
    _cfg0 = R._cfg
    R._cfg = lambda: dataclasses.replace(_cfg0(), species_variables=[s + "_evolution" for s in SPECIES_ALL])


def kind(g):
    """Start type: the group's 'start' attr (for design rows it equals the design's kind)."""
    return str(g.attrs["start"])


def jobs(K, n_per):
    out = []
    files = FILES.split(",") if FILES else [f"run_gcm_b{b:02d}-result.h5" for b in range(K)]
    assert len(files) == K, (files, K)
    for fn in files:
        fp = RAW / fn
        with h5py.File(fp, "r") as f:
            assert SPECIES_ALL is None or pack_species(f) == SPECIES_ALL, f"{fp}: species differ from the first file's"
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


def merge(K, n_per):
    res = [pickle.load(open(R.CACHE_DIR / f"part_{i}.pkl", "rb")) for i in range(K * n_per)]
    files = sorted({r[0] for r in res})
    t_grid = res[0][3]
    rel = max(float(np.max(np.abs(r[3] / t_grid - 1)[1:])) for r in res)
    assert rel < 1e-12, rel
    rej, counts = {}, {}
    for r in res:
        for k, v in r[2].items():
            rej[k] = rej.get(k, 0) + v
    for sp in R.SPLITS:
        o = [(r[0], r[1][sp]) for r in res if r[1][sp]["end"]]
        cat = lambda k, dt: np.concatenate([np.asarray(x[k], dtype=dt) for _, x in o])
        fidx = np.concatenate([np.full(len(x["end"]), files.index(fn), np.int8) for fn, x in o])
        np.savez(R.CACHE_DIR / f"raw_{sp}.npz", logy=np.concatenate([np.stack(x["logy"]) for _, x in o]),
                 end=cat("end", np.int16), conv=cat("conv", np.int16), P=cat("P", np.float64), T=cat("T", np.float64),
                 seed=cat("seed", np.int64), file_idx=fidx, lo=np.full(fidx.size, -25.0), name=cat("name", "U64"))
        counts[sp] = int(fidx.size)
    np.save(R.CACHE_DIR / "t_grid.npy", t_grid)
    meta = {"files": files, "kept_by_split": counts, "kept_total": sum(counts.values()), "group_rejects": rej,
            "grid_rel_diff": rel, "design_rows": K * 2 ** 16}
    (R.CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    for i in range(K * n_per):
        (R.CACHE_DIR / f"part_{i}.pkl").unlink()


if __name__ == "__main__":
    a = sys.argv[1:]
    chunk(int(a[1]), int(a[2]), int(a[3])) if a[0] == "chunk" else merge(int(a[1]), int(a[2]))
