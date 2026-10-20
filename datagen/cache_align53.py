"""A 53-species cache paired exactly with a 12-species gcm cache (building the 53-species cache through rawtraj's own
filters would give species-dependent frozen-tail cuts and rejections, so trajectories and their order would differ). Same groups, order, split, end and conv rows, P, T, seed as the 12-species cache; logy = log10 max(y,
1e-30) of every network species (the raw files' 'species' attr: the generator's order, He included) read from the
same raw files; no generator import. Checks that the 12
shared columns equal the 12-species cache exactly.

    python cache_align53.py <12-species cache dir> <out dir> [<raw dir>, default paths.RAW_GCM]
"""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

EPS = 1e-30  # production epsilon, as the 12-species cache
src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / "meta.json").read_text())
files = meta["files"]
raw = Path(sys.argv[3]) if len(sys.argv) > 3 else paths.RAW_GCM
handles = {k: h5py.File(raw / f, "r") for k, f in enumerate(files)}
sps = [[s.decode() for s in h.attrs["species"]] for h in handles.values()]
assert all(s == sps[0] for s in sps), "raw files list different species"
sp = sps[0]
for split in ("train", "validation", "test"):
    d = dict(np.load(src / f"raw_{split}.npz"))
    names, fidx = d["name"], d["file_idx"]
    logy = np.empty(d["logy"].shape[:2] + (len(sp),), dtype=np.float64)
    for k, (nm, fi) in enumerate(zip(names, fidx)):
        g = handles[int(fi)][str(nm)]
        logy[k] = np.log10(np.maximum(np.stack([g[s + "_evolution"][()] for s in sp], -1), EPS))
    # the 12-species cache stores its columns in the production species order (data.SHORT); compare by name
    norm = json.loads(paths.NORM12.read_text())
    short = [s.replace("_evolution", "") for s in norm["species_variables"]]
    cols = [sp.index(s) for s in short]
    assert np.array_equal(logy[:, :, cols], d["logy"]), f"{split}: tracked columns differ from the 12-species cache"
    d["logy"] = logy
    np.savez(out / f"raw_{split}.npz", **d)
    print(f"{split}: {len(names)} trajectories, 53 species, tracked columns identical", flush=True)
shutil.copy(src / "t_grid.npy", out / "t_grid.npy")
meta.update({"species": sp, "aligned_to": str(src), "note": "cache_align53.py: same groups, order, end/conv as aligned_to"})
(out / "meta.json").write_text(json.dumps(meta, indent=1))
