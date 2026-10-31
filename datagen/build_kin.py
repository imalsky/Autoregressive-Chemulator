"""Training set of the kinetic solver: VULCAN clean-design runs mapped onto a reduced species set.

For the chosen rows of the clean design (data/gen/clean/design_map.npy; default: GCM-occupancy and cooler strata, Z 0.5-2,
C/O 0.4-0.75, perturbed starts up to sigma 1 dex), every stored state (100 times, 0 to 1e8 s) is mapped onto the reduced
species (map_reduced) and written with T, P, start kind and the elements per H atom.

map_reduced: the reduced species keep their own mixing ratios; the C, N, O, H and He atoms held in the other species are
placed: C into CO or CH4, whichever is larger; the O balance (hidden O minus the O of that CO) into H2O; N into NH3 or N2,
whichever is larger; He into He; the H balance into H2. H2O and H2 are clipped at 0 if a balance would make them
negative (the H shortfall is reported). Elements are kept exactly unless a clip occurs.
    $PY datagen/build_kin.py <out.npz> [n_workers]
"""
import re
import sys
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

RED = ["OH", "H2", "H2O", "H", "CO", "CO2", "O", "CH4", "C2H2", "NH3", "N2", "HCN", "He"]
ELEM = ["C", "H", "He", "N", "O"]
SEED0 = 4210000000


def atoms(names):
    cnt = []
    for s in names:   # element counts, summed over repeats (CH3OH: H 4)
        c = {}
        for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", s):
            c[e] = c.get(e, 0) + int(n or 1)
        cnt.append(c)
    return np.array([[c.get(e, 0) for e in ELEM] for c in cnt], float)


def map_reduced(X, names):
    """X [n, len(names)] mixing ratios -> ([n, len(RED)] mixing ratios summing to 1, H shortfall per row)."""
    idx = [names.index(s) for s in RED]
    hid = np.ones(len(names), bool)
    hid[idx] = False
    A = atoms(names)
    h = X[:, hid] @ A[hid]                                     # hidden atoms per row: C H He N O
    hC, hH, hHe, hN, hO = h.T
    v = X[:, idx].copy()
    k = {s: i for i, s in enumerate(RED)}
    co = np.where(v[:, k["CO"]] >= v[:, k["CH4"]], hC, 0.0)
    v[:, k["CO"]] += co
    v[:, k["CH4"]] += hC - co
    h2o = np.maximum(v[:, k["H2O"]] + hO - co, 0.0)
    dh2o = h2o - v[:, k["H2O"]]
    v[:, k["H2O"]] = h2o
    to_nh3 = v[:, k["NH3"]] >= v[:, k["N2"]]
    v[:, k["NH3"]] += np.where(to_nh3, hN, 0.0)
    v[:, k["N2"]] += np.where(to_nh3, 0.0, hN / 2.0)
    v[:, k["He"]] += hHe
    h_used = 4.0 * (hC - co) + 2.0 * dh2o + np.where(to_nh3, 3.0 * hN, 0.0)
    h2 = v[:, k["H2"]] + (hH - h_used) / 2.0
    short = np.maximum(-h2, 0.0)
    v[:, k["H2"]] = np.maximum(h2, 0.0)
    return v / v.sum(1, keepdims=True), short


def one(args):
    block, keys = args
    out = []
    with h5py.File(paths.RAW_CLEAN / f"run_clean_b{block:02d}-result.h5", "r") as f:
        names = [s.decode() for s in f.attrs["species"]]
        for row, key in keys:
            g = f[key]
            X = np.stack([g[f"{s}_evolution"][()] for s in names], 1)
            X = X / X.sum(1, keepdims=True)
            v, short = map_reduced(X, names)
            out.append((row, g["t_time"][()], v, short.max()))
    return out


def pick(d):
    return np.flatnonzero(np.isin(d["stratum"], ["G", "C"]) & (d["Z"] >= 0.5) & (d["Z"] <= 2) & (d["C_O"] >= 0.4)
                          & (d["C_O"] <= 0.75) & (np.isnan(d["sigma"]) | (d["sigma"] <= 1)))


if __name__ == "__main__":
    out_path, nw = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = np.asarray(np.load(paths.DATA / "gen/clean/design_map.npy", mmap_mode="r")[:786432])
    rows = pick(d)
    jobs = []
    for b in sorted({int(r) // 65536 for r in rows}):
        with h5py.File(paths.RAW_CLEAN / f"run_clean_b{b:02d}-result.h5", "r") as f:
            have = {int(k.rsplit("_", 1)[1]) - SEED0: k for k in f.keys()}
        mine = [(int(r), have[int(r)]) for r in rows if int(r) // 65536 == b and int(r) in have]
        jobs += [(b, mine[i:i + 500]) for i in range(0, len(mine), 500)]
    with Pool(nw) as p:
        res = [x for part in p.imap(one, jobs) for x in part]
    res.sort(key=lambda x: x[0])
    r = np.array([x[0] for x in res])
    np.savez(out_path, row=r, t=res[0][1], x=np.array([x[2] for x in res], np.float64),
             short=np.array([x[3] for x in res]), species=np.array(RED), T=d["T"][r], Pbar=d["Pbar"][r],
             kind=d["kind"][r], stratum=d["stratum"][r], Z=d["Z"][r], C_O=d["C_O"][r], He_H=d["He_H"][r])
    print(f"{len(r)} runs of {len(rows)} picked; largest H shortfall {max(x[3] for x in res):.2e}")
