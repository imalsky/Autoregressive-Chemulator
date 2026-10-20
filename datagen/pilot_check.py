"""Arm M pilot checks (plan section 5, M pilot), on the 12 tracked species, resolved points only (number density >= 1e-2
cm^-3 at the start or at the time compared):
 (a) zero-shift controls stay put: max over t <= 500 s of |log10 y(t) - log10 y(0)| (target <= 1e-3 dex);
 (b) tolerance: per row, the move over the first 500 s at rtol 0.05 vs the |rtol 0.05 - rtol 0.005| difference at 500 s;
     the 1-5 K rows must move >= 3x that difference;
 (c) what the rows teach: per band, the one-call (10, 100, 500 s) move from the start and the share of the start's
     displacement from the final state that is gone after one call (the relaxation fraction the model should learn).
    python pilot_check.py [set]      (default pilot)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DATAGEN))
import generate as G  # noqa: E402
import gcm  # noqa: E402

HERE = paths.GEN_SHIFT   # gen_shift sets (2026-10-09: was this script's folder)
S = sys.argv[1] if len(sys.argv) > 1 else "pilot"
KB = 1.380649e-16
SHORT = ["C2H2", "CH4", "CO", "CO2", "H2", "H2O", "H", "HCN", "N2", "NH3", "O", "OH"]
sp = gcm.species()
cols = [sp.index(s) for s in SHORT]
lt = np.log10(G.T_GRID[1:])
m = dict(np.load(HERE / S / "manifest.npz"))
EDGES = [300, 800, 1300, 1800, 2300, 3001]


def ly_at(grid, t):
    """log10 y [12] at time t (s): row 0 at t = 0, else log-log interpolation between the positive rows (the harness's)."""
    y = np.log10(np.maximum(grid[:, cols], 1e-30))
    if t == 0:
        return y[0]
    return np.array([np.interp(np.log10(t), lt, y[1:, j]) for j in range(len(cols))])


def load(rtol, k):
    f = HERE / S / f"boxes_{rtol}" / f"{k}.npz"
    return np.load(f) if f.exists() else None


rows = []
for j, k in enumerate(m["k"]):
    a, b = load("0.05", int(k)), load("0.005", int(k))
    if a is None:
        continue
    T, P = gcm.params(int(m["i"][j]))["T"], gcm.params(int(m["i"][j]))["P"]
    lnd = np.log10(P / (KB * T))
    y0 = ly_at(a["grid"], 0.0)
    yend = np.log10(np.maximum(a["grid"][-1, cols], 1e-30))
    rec = {"k": int(k), "mode": int(m["mode"][j]), "T": T, "dT": float(m["dT"][j]), "dlogP": float(m["dlogP"][j]),
           "x0_ok": float(np.max(np.abs(np.log10(np.maximum(a["grid"][0], 1e-300)) - np.log10(np.maximum(a["x0"] / a["x0"].sum(), 1e-300)))))}
    res0 = (y0 + lnd) >= -2.0
    for t in (10.0, 100.0, 500.0):
        yt = ly_at(a["grid"], t)
        r = res0 | ((yt + lnd) >= -2.0)
        rec[f"move{t:g}"] = float(np.max(np.abs(yt - y0)[r])) if r.any() else 0.0
        disp = np.abs(y0 - yend)
        big = r & (disp > 3e-3)
        rec[f"frac{t:g}"] = float(np.median(1.0 - np.abs(yt - yend)[big] / disp[big])) if big.any() else np.nan
    tq = [t for t in G.T_GRID if 0 < t <= 500.0]
    rec["drift500"] = float(max(np.max(np.abs(ly_at(a["grid"], t) - y0)[res0]) for t in tq))
    rec["disp"] = float(np.max(np.abs(y0 - yend)[res0 | ((yend + lnd) >= -2.0)]))
    if b is not None:
        r = res0 | ((ly_at(b["grid"], 500.0) + lnd) >= -2.0)
        rec["rtoldiff500"] = float(np.max(np.abs(ly_at(a["grid"], 500.0) - ly_at(b["grid"], 500.0))[r]))
        rec["move500_tight"] = float(np.max(np.abs(ly_at(b["grid"], 500.0) - ly_at(b["grid"], 0.0))[r]))
    rows.append(rec)

R = {key: np.array([r.get(key, np.nan) for r in rows]) for key in rows[0]}
print(f"set {S}: {len(rows)} rows with an rtol 0.05 box, {np.isfinite(R['rtoldiff500']).sum()} with both")
print(f"row 0 = solver x0 (normalized): max |dlog10| {np.nanmax(R['x0_ok']):.1e}")
z = R["mode"] == 3
print(f"(a) zero-shift controls ({z.sum()}): drift over t <= 500 s p50 {np.median(R['drift500'][z]):.2e}, p90 "
      f"{np.percentile(R['drift500'][z], 90):.2e}, max {R['drift500'][z].max():.2e} dex (target <= 1e-3); final-state offset "
      f"from the start p50 {np.median(R['disp'][z]):.2e} max {R['disp'][z].max():.2e}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    mm = z & (R["T"] >= lo) & (R["T"] < hi)
    if mm.any():
        print(f"    {lo}-{hi} K: n {mm.sum()}, drift max {R['drift500'][mm].max():.2e}, offset max {R['disp'][mm].max():.2e}")
small = (R["mode"] == 0) & (np.abs(R["dT"]) <= 5.0) & np.isfinite(R["rtoldiff500"])
ratio = R["move500"][small] / np.maximum(R["rtoldiff500"][small], 1e-12)
print(f"(b) 1-5 K T-only rows ({small.sum()}): move over 500 s p50 {np.median(R['move500'][small]):.2e}; |rtol 0.05 - 0.005| "
      f"at 500 s p50 {np.median(R['rtoldiff500'][small]):.2e}; ratio p10 {np.percentile(ratio, 10):.2f} p50 {np.median(ratio):.2f} "
      f"(need >= 3)")
sh = np.logical_not(z) & np.isfinite(R["rtoldiff500"])
print(f"    all shifted rows: |rtol diff| at 500 s p50 {np.median(R['rtoldiff500'][sh]):.2e} p90 {np.percentile(R['rtoldiff500'][sh], 90):.2e}; "
      f"move p50 {np.median(R['move500'][sh]):.2e}")
print("(c) shifted rows per band: start displacement from the final state p50; move after 10 / 100 / 500 s p50; median "
      "share of the displacement gone after 10 / 100 / 500 s")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    mm = np.logical_not(z) & (R["T"] >= lo) & (R["T"] < hi)
    if mm.any():
        print(f"    {lo}-{hi} K (n {mm.sum()}): disp {np.median(R['disp'][mm]):.3f}; move {np.median(R['move10'][mm]):.3f} / "
              f"{np.median(R['move100'][mm]):.3f} / {np.median(R['move500'][mm]):.3f}; gone {np.nanmedian(R['frac10'][mm]):.2f} / "
              f"{np.nanmedian(R['frac100'][mm]):.2f} / {np.nanmedian(R['frac500'][mm]):.2f}")
