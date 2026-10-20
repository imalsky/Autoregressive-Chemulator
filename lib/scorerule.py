"""Which points count and their error, for every board (2026-10-11). SCORE_RULE unset or "flat12": the 10-08 rule
(resolved points with truth or prediction >= 1e-12 mixing ratio; error |pred - truth| in dex). SCORE_RULE=phys: the
author's per-gas rule of 2026-10-11 (project CLAUDE.md "Scoring rule"): floors S per gas, a resolved point counts if
truth or prediction >= S, error |max(pred, S) - max(truth, S)|, so an invented gas counts by how far it rises above S."""
import os

import numpy as np

RULE = os.environ.get("SCORE_RULE", "flat12")
assert RULE in ("flat12", "phys"), RULE
FLOOR = -12.0
PHYS = {"H2O": -10.0, "CO": -10.0, "CO2": -10.0, "CH4": -10.0, "NH3": -10.0, "HCN": -10.0, "C2H2": -10.0,
        "H": -8.0, "OH": -8.0, "O": -8.0, "H2": -6.0, "N2": -6.0}


def err(vt, st, res, names):
    """vt, st: log10 truth and prediction [..., 12]; res: the resolved (number density) mask, same shape; names: the
    12 species in the last axis's order. Returns the error on scored points, NaN elsewhere."""
    if RULE == "flat12":
        return np.where(res & ((vt >= FLOOR) | (st >= FLOOR)), np.abs(st - vt), np.nan)
    L = np.array([PHYS[s] for s in names])
    return np.where(res & ((vt >= L) | (st >= L)), np.abs(np.maximum(st, L) - np.maximum(vt, L)), np.nan)
