"""Shared E13 definitions (from the 2026-10-13 failure diagnosis; used by build_kicks.py). Read-only on every input.
Truth sets, the 25-entry definition (datagen/cache_res.py L28-31: 12 reported + He + 7 named tracked,
reservoirs C/N/O/H atoms and molecule count over the 33 others), per-gas floors, switch timing."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

# truth set -> (truth tag dir, None); the second slot held the F2 bench of the 10-13 diagnosis (not used here)
SETS = {"G": (paths.GTEST, None), "Gc": (paths.GTEST_COLD, None), "H2": (paths.HOLDOUTS["h2"], None),
        "H2c": (paths.HOLDOUTS["h2c"], None), "H3": (paths.HOLDOUTS["h3"], None)}
SP12 = ["C2H2", "CH4", "CO2", "CO", "H2O", "H2", "HCN", "H", "N2", "NH3", "OH", "O"]
TRACKED = SP12 + ["He", "C", "N", "CH3", "H2CN", "C2H6", "CN", "O2"]
RES = ["C_res", "N_res", "O_res", "H_res", "M_res"]
NAMES25 = TRACKED + RES
S = {"H2O": -10.0, "CO": -10.0, "CO2": -10.0, "CH4": -10.0, "NH3": -10.0, "HCN": -10.0, "C2H2": -10.0,
     "H": -8.0, "OH": -8.0, "O": -8.0, "H2": -6.0, "N2": -6.0}
L12 = np.array([S[s] for s in SP12])
KB = 1.380649e-16
CALLS = np.array([1, 2, 3, 6, 11, 19, 33, 58, 103, 183, 323, 572, 1013, 1792, 3173, 5616, 9942, 17599, 31153, 55146,
                  97618, 155520, 172800])
T_END = 500.0 * 172800    # 1000 days, the benches' horizon


def atoms(sp):
    n = lambda e: sum(int(k or 1) for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", sp) if el == e)
    return np.array([n("C"), n("N"), n("O"), n("H"), 1.0])     # C, N, O, H, molecule


def wmat(species):
    """[53, 5] atoms per molecule (C, N, O, H, 1) and the hidden mask (not in the 20 tracked + He)."""
    W = np.stack([atoms(s) for s in species])
    hid = np.array([s not in TRACKED for s in species])
    return W, hid


def to25(x, species):
    """linear 53-species x [..., 53] -> linear 25-entry state."""
    W, hid = wmat(species)
    return np.concatenate([x[..., [species.index(s) for s in TRACKED]], x[..., hid] @ W[hid]], -1)


def job(set_, j):
    d, _ = SETS[set_]
    return json.loads((d / "jobs.json").read_text())[j]


def truth(set_, j):
    d, _ = SETS[set_]
    r = np.load(d / f"{j}.npz")
    return r["t"], r["x"], [str(s) for s in r["species"]]


def interp(t, x, times):
    """linear-in-t interpolation of linear x (score.vtruth's rule) -> [len(times), S]"""
    return np.stack([np.interp(times, t, x[:, k]) for k in range(x.shape[1])], -1)


def l12(t, x, species, times):
    xi = interp(t, x, times)
    return np.log10(np.maximum(xi[:, [species.index(s) for s in SP12]], 1e-30))


def perr(la, lb, resolved=None):
    """per-gas rule error between two log10 12-species arrays [..., 12]; NaN where neither >= S (or unresolved)."""
    ok = (la >= L12) | (lb >= L12)
    if resolved is not None:
        ok &= resolved
    return np.where(ok, np.abs(np.maximum(la, L12) - np.maximum(lb, L12)), np.nan)


def resolved(la, lb, P, T):
    lnd = np.log10(P / (KB * T))
    return ((la + lnd) >= -2.0) | ((lb + lnd) >= -2.0)


def switch_time(t, y, t_lo, t_hi, frac=0.5):
    """time at which log10 y (floored series sampled on the run's own steps, absolute t) first crosses
    a + frac (b - a) between its values at t_lo and t_hi; NaN if |b - a| < 0.5 dex. Log-linear interpolation of
    the crossing on a fine log grid."""
    tt = np.logspace(np.log10(t_lo), np.log10(t_hi), 4001)
    yy = np.interp(np.log10(tt), np.log10(np.maximum(t, 1e-300)), y)
    a, b = yy[0], yy[-1]
    if abs(b - a) < 0.5:
        return np.nan
    lev = a + frac * (b - a)
    hit = np.flatnonzero((yy - lev) * np.sign(b - a) >= 0)
    return float(tt[hit[0]]) if hit.size else np.nan
