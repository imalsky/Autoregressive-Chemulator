"""A reservoir-set cache and its normalization from the 53-species cache (same groups, order, split, end/conv,
P, T, seed). Entries: the 12 reported species (production order), He, the 7 species a greedy element-coverage pass
added first on eq/mix states (C, N, CH3, H2CN, C2H6, CN, O2), and 5 reservoirs over the remaining 33 species: C_res,
N_res, O_res, H_res (atoms of each element) and M_res (molecule count). log10 max(value, 1e-30). Element ratios to H
and sum 1 over molecules (tracked + M_res) equal those of the full 53-species state. The normalization is the
12-species file with the species entries replaced by training-split statistics (log_std floored at 0.5).

    python cache_res.py <53-species cache dir> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import elemproj as EP  # noqa: E402

EPS = 1e-30
SNAP = paths.NORM12
src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / "meta.json").read_text())
sp = list(meta["species"])
norm = json.loads(SNAP.read_text())
short12 = [s.replace("_evolution", "") for s in norm["species_variables"]]
tracked = short12 + ["He", "C", "N", "CH3", "H2CN", "C2H6", "CN", "O2"]
RES = ["C_res", "N_res", "O_res", "H_res", "M_res"]
names = tracked + RES
h, a = (t.numpy() for t in EP.stoich(sp))
hid = np.array([s not in tracked for s in sp])
W = np.stack([a[:, 0], a[:, 1], a[:, 2], h, np.ones(len(sp))], -1)[hid]       # [33, 5]
cols = [sp.index(s) for s in tracked]
stats = None
for split in ("train", "validation", "test"):
    d = dict(np.load(src / f"raw_{split}.npz"))
    y = 10.0 ** d["logy"]                                                        # [N, 100, 53]
    v = np.concatenate([y[..., cols], y[..., hid] @ W], -1)
    d["logy"] = np.log10(np.maximum(v, EPS))
    # check: element ratios to H and the molecular sum agree with the full state
    full = np.stack([y @ a[:, 0], y @ a[:, 1], y @ a[:, 2]], -1) / (y @ h)[..., None]
    hr, ar = (t.numpy() for t in EP.stoich(names))
    mol = EP.molidx(names).numpy()
    part = np.stack([v @ ar[:, 0], v @ ar[:, 1], v @ ar[:, 2]], -1) / (v @ hr)[..., None]
    rel = np.abs(part / full - 1).max()
    msum = np.abs(v[..., mol].sum(-1) / y.sum(-1) - 1).max()
    print(f"{split}: {v.shape[0]} trajectories, {len(names)} entries; max rel diff of C,N,O/H {rel:.2e}, of the molecular sum {msum:.2e}", flush=True)
    assert rel < 1e-9 and msum < 1e-9
    np.savez(out / f"raw_{split}.npz", **d)
    if split == "train":
        m = np.arange(d["logy"].shape[1])[None, :] <= d["end"][:, None]
        pts = d["logy"][m]
        stats = (pts.mean(0), np.maximum(pts.std(0), 0.5), pts.shape[0])
shutil.copy(src / "t_grid.npy", out / "t_grid.npy")
meta.update({"species": names, "derived_from": str(src), "note": "cache_res.py: reservoir set"})
(out / "meta.json").write_text(json.dumps(meta, indent=1))
norm["species_variables"] = [s + "_evolution" for s in names]
for k, s in enumerate(norm["species_variables"]):
    norm["per_key_stats"][s] = {**norm["per_key_stats"].get(s, {}), "log_mean": float(stats[0][k]), "log_std": float(stats[1][k])}
norm["e13_note"] = f"species stats from {out} raw_train.npz ({stats[2]} grid points); other entries from {SNAP}"
(out / "norm.json").write_text(json.dumps(norm, indent=1))
print(f"wrote {out}/norm.json: log_std range {stats[1].min():.3g}-{stats[1].max():.3g}")
