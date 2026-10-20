"""Per-row draw weights of CBg's training cache U4 (w_U4_s1, w_U4_s2; train.py traj_weights_file). Note: its second
argument, the U2 reservoir cache ($CHEM_SCRATCH/arch_gcmU2_res), no longer exists; until it is rebuilt from the raw packs
this script cannot regenerate the weight files.
U4 = the U2 prefix + y21b + kicks + n1 + n2; every row after the U2 prefix that is not a kick joins its stratum like y21b.
Optional 6th argument: the kicked share (default 0.20; 0 for a stage-1 file; old strata then get 1 - that share).
The rule (written for U3 = U2 + y21b + kicks; U4 adds n1, n2 as above):
Strata (strata() below): six T bands (0-650-800-1300-1800-2300-inf K) x
event flag for non-M rows, plus the M rows x event flag (14 strata); the kicked rows (run_gcm_z22k) form four more strata
by band (300-650, 650-1300, 1300-2000, 2000-3000 K).
  old strata: FINAL's realised stage-2 shares (bincount over U2 train rows with w_U2_s2) x 0.80, spread uniformly over the
    stratum's usable U3 rows (the old rows plus the y21b rows that fall in it: y21b is 650-1000 K, so it joins the 650-800
    and 800-1300 K strata, event flag recomputed by the same rule);
  kick strata: 0.20 x (0.30, 0.45, 0.15, 0.10), uniform over the band's usable kicked rows.
Usable: old rows w_U2_s2 > 0 (U3's train prefix is U2's train, asserted here on seed and name and, for the event flags,
on the strata themselves); new rows elem_drift <= 1e-2, read from the raw groups' attrs (y21b and z22k).
    python weights_u4.py <U4 res cache> <U2 res cache> <w_U2_s2.npy> <U4 raw dir> <out .npy> [kicked share]
                                                                                              (refuses to overwrite)"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

cache, cache2, w2f, raw, out = (Path(a) for a in sys.argv[1:6])
if out.exists():
    raise FileExistsError(out)
EDG = np.array([0.0, 650.0, 800.0, 1300.0, 1800.0, 2300.0, 1e9])
FL = {"CH4": -10.0, "C2H2": -10.0, "HCN": -10.0, "H": -8.0}
KB = ((300.0, 650.0), (650.0, 1300.0), (1300.0, 2000.0), (2000.0, 3000.0))
KSH = (0.30, 0.45, 0.15, 0.10)
KICK_SHARE = float(sys.argv[6]) if len(sys.argv) > 6 else 0.20
OLD_SHARE = 1.0 - KICK_SHARE


def strata(c):
    """Per train row: stratum index (band * 2 + event for non-M rows, 12 + event for M
    rows); also returns the cache arrays."""
    meta = json.loads((c / "meta.json").read_text())
    files, spc = meta["files"], meta["species"]
    z = np.load(c / "raw_train.npz")
    tg = np.load(c / "t_grid.npy")
    T, end = z["T"], z["end"]
    m = np.isin(z["file_idx"], [k for k, f in enumerate(files) if f.startswith("run_gcm_m18")])
    i3 = int(np.searchsorted(tg[1:], 1e3))
    ci = np.flatnonzero(T < 1000.0)
    cols = [spc.index(s) for s in FL]
    ly = np.maximum(z["logy"][ci][:, 1:, :][:, :, cols], np.array(list(FL.values()))[None, None, :])
    valid = np.arange(1, 100)[None, :] <= end[ci][:, None]
    sw = np.zeros((ci.size, len(FL)))
    for i in range(i3, 98):
        if i + 9 < 99:
            sw = np.maximum(sw, np.where(valid[:, i + 9][:, None], np.abs(ly[:, i + 9] - ly[:, i]), 0.0))
    ev = np.zeros(T.size, bool)
    ev[ci[(sw > 2.0).any(1)]] = True
    s = (np.digitize(T, EDG) - 1) * 2 + ev
    s[m] = 12 + ev[m]
    return s, z, files


s2, z2, f2 = strata(cache2)
w2 = np.load(w2f).astype(np.float64)
assert w2.size == s2.size
share = np.bincount(s2, weights=w2, minlength=14) / w2.sum()
s3, z3, f3 = strata(cache)
n2 = s2.size
seed3, name3, fidx3, T3 = z3["seed"], z3["name"], z3["file_idx"], z3["T"]
assert f3[:len(f2)] == f2
assert np.array_equal(seed3[:n2], z2["seed"]) and np.array_equal(name3[:n2], z2["name"]), "U3 train prefix != U2 train"
assert np.array_equal(s3[:n2], s2), "strata of the U2 prefix differ"
assert np.all(fidx3[n2:] >= len(f2)), "new rows not after the old ones"
usable = np.ones(s3.size, bool)
usable[:n2] = w2 > 0
for k in range(len(f2), len(f3)):
    idx = np.flatnonzero(fidx3 == k)
    nm = name3[idx]
    with h5py.File(raw / f3[k], "r") as h:
        dr = np.array([float(h[str(q)].attrs["elem_drift"]) for q in nm])
    usable[idx] = dr <= 1e-2
    print(f"{f3[k]}: {idx.size} train rows, elem_drift > 1e-2: {int((dr > 1e-2).sum())} (max {dr.max():.3g})")
kick = fidx3 == f3.index("run_gcm_z22k-result.h5")
y21b = (fidx3 >= len(f2)) & np.logical_not(kick)
for b, (lo, hi) in enumerate(KB):
    sel = kick & (T3 >= lo) & ((T3 < hi) if hi < 3000 else (T3 <= hi))
    s3[sel] = 14 + b
assert np.all(s3[kick] >= 14), "a kicked row outside 300-3000 K"
w = np.zeros(s3.size)
tgt = np.concatenate([OLD_SHARE * share, KICK_SHARE * np.array(KSH)])
for k in range(18):
    sel = (s3 == k) & usable
    if tgt[k] > 0:
        assert sel.any(), f"stratum {k} has target share {tgt[k]:.4f} but no usable rows"
        w[sel] = tgt[k] / sel.sum()
new = np.bincount(s3, weights=w, minlength=18) / w.sum()
n_u2 = np.bincount(s2[w2 > 0], minlength=14)
n_u3 = np.bincount(s3[usable], minlength=18)
n_y = np.bincount(s3[y21b & usable], minlength=18)
for k in range(18):
    if k < 12:
        lab = f"T {EDG[k // 2]:g}-{EDG[k // 2 + 1]:g} K {'event' if k % 2 else 'other'}"
    elif k < 14:
        lab = f"M rows {'event' if k == 13 else 'other'}"
    else:
        lab = f"KICK {KB[k - 14][0]:g}-{KB[k - 14][1]:g} K"
    old = f"share U2 {share[k]:.4f}, " if k < 14 else ""
    rel = (f"; per-row draw rate vs U2 x{(new[k] / max(n_u3[k], 1)) / (share[k] / max(n_u2[k], 1)):.2f}"
           if k < 14 and share[k] > 0 else "")
    print(f"{lab:28s} {old}U3 {new[k]:.4f} (target {tgt[k]:.4f}); usable rows U2 {n_u2[k] if k < 14 else 0} U3 {n_u3[k]} "
          f"(y21b {n_y[k]}){rel}")
print(f"train rows {s3.size} (U2 prefix {n2}, y21b {int(y21b.sum())}, kicked {int(kick.sum())}); zero weight {int((w == 0).sum())}; "
      f"max |share - target| {np.abs(new - tgt).max():.2e}; kicked share {new[14:].sum():.4f}")
assert np.abs(new - tgt).max() < 1e-9
np.save(out, w)
print(f"wrote {out}")
