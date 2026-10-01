#!/usr/bin/env python3
"""make_synthetic_raw.py — tiny synthetic raw HDF5 files in the VULCAN schema.

Purpose: local end-to-end smoke testing of preprocessing + training configs
(see testing/run_local_smoke.sh) on machines that do not hold the
real raw dataset. NOT for science: species curves are smooth synthetic decays.

Schema produced (the layout of the real raw files in data/raw, checked on daw 2026-09-30):
  file.h5
    run_NNN_NNNN/              one group per trajectory
      t_time                   [T] float64: t = 0, then T-1 log-spaced times (the real save grid)
      <species>_evolution      [T] float64 mixing ratios in (0, 1]; the t=0 composition sums to 1
      attrs: P (dyn/cm^2), T (K) float64 scalars; SEED int64
  The first file also holds the two reject cases the real data contains (preprocessing counts
  them in preprocessing_summary.json group_rejects; testing/run_local_smoke.sh asserts both):
    ZERO_GROUP   exact zeros in one species at t > 0 (drop_below)
    NAN_GROUP    NaN t_time and species, finite P/T attrs (non_finite)

Usage:
    AUTOCHEM_SYNTH_OUT_DIR=smoke_scratch/raw python processing/make_synthetic_raw.py
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

SPECIES = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution",
]

# Generation settings (no argparse; module-level constants per spec 9.3.4): the time grid, the
# per-trajectory distributions and the run settings below. Smoke use only: not traced to the real
# data except where noted. The output directory is the one run-varying input and comes from a required
# environment variable (mirrors the AUTOCHEM_CONFIG_PATH fail-fast convention).
# The real save grid: t = 0 then 99 log-spaced times from 1e-3 s to 1e8 s.
T_FIRST_POSITIVE_S, T_LAST_S = 1e-3, 1e8
T_POSITIVE_POINTS = 99
# Per-species curves y(t) = 10 ** (log_eq + (log_init - log_eq) * exp(-t / tau)), with the t=0
# levels then rescaled so the initial composition sums to 1 (the real initial gas). Each curve moves
# monotonically in log space between its t=0 level (<= 1) and log_eq (< 0), so every value stays
# <= 1; the 12-species sum after t=0 is not constrained (it can exceed 1, unlike the real data).
LOG_EQ_RANGE = (-20.0, -0.5)        # log10 of the equilibrium level, uniform per species
LOG_INIT_OFFSET_DEX = 3.0           # log10 initial level (before rescaling) = log_eq + U(-this, this)
LOG_TAU_RANGE_S = (-2.0, 4.0)       # log10 of the relaxation time (s), uniform per species
# Per-trajectory globals, in the real data's units and ranges.
LOG_P_RANGE_BARYE = (0.0, 10.0)     # log10 P (dyn/cm^2), uniform: P in 1..1e10 like the real data
T_RANGE_K = (300.0, 3000.0)         # T (K), uniform; matches the real 300-3000 K envelope
SEED_ATTR_HIGH = 2**32              # SEED attrs are 32-bit generator seeds (real values ~2.2e9, int64)
# Reject cases (first file only).
ZERO_GROUP = "run_000_zeros"
ZERO_SPECIES_INDEX = 0              # this species is exactly 0 at every t > 0
NAN_GROUP = "run_000_nan"
# Run settings.
OUT_DIR_ENV_VAR = "AUTOCHEM_SYNTH_OUT_DIR"
N_FILES = 2
N_TRAJ = 30  # regular trajectories per file
SEED = 0


def synth_trajectory(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    t = np.concatenate(([0.0], np.logspace(np.log10(T_FIRST_POSITIVE_S), np.log10(T_LAST_S), T_POSITIVE_POINTS)))
    s = len(SPECIES)
    # Smooth positive curves: equilibrium level + decaying component, in log10 space.
    log_eq = rng.uniform(*LOG_EQ_RANGE, size=s)
    log_init = log_eq + rng.uniform(-LOG_INIT_OFFSET_DEX, LOG_INIT_OFFSET_DEX, size=s)
    log_init -= np.log10(np.sum(10.0 ** log_init))  # the t=0 composition sums to 1
    tau = 10.0 ** rng.uniform(*LOG_TAU_RANGE_S, size=s)
    decay = np.exp(-t[:, None] / tau[None, :])
    log_y = log_eq[None, :] + (log_init - log_eq)[None, :] * decay
    y = 10.0 ** log_y
    P = float(10.0 ** rng.uniform(*LOG_P_RANGE_BARYE))
    T_K = float(rng.uniform(*T_RANGE_K))
    seed_attr = int(rng.integers(0, SEED_ATTR_HIGH))
    return t, y, P, T_K, seed_attr


def write_group(f: h5py.File, name: str, t: np.ndarray, y: np.ndarray, P: float, T_K: float, seed_attr: int) -> None:
    grp = f.create_group(name)
    grp.create_dataset("t_time", data=t)
    for j, species in enumerate(SPECIES):
        grp.create_dataset(species, data=y[:, j])
    grp.attrs["P"] = P
    grp.attrs["T"] = T_K
    grp.attrs["SEED"] = np.int64(seed_attr)


def _resolve_out_dir() -> Path:
    raw = os.environ.get(OUT_DIR_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{OUT_DIR_ENV_VAR} is not set. Point it at the output directory, e.g.:\n"
            f"  {OUT_DIR_ENV_VAR}=smoke_scratch/raw python processing/make_synthetic_raw.py"
        )
    return Path(raw).expanduser()


def main() -> None:
    out = _resolve_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for fi in range(N_FILES):
        path = out / f"synthetic_run{fi:03d}.h5"
        with h5py.File(path, "w") as f:
            for ti in range(N_TRAJ):
                write_group(f, f"run_{fi:03d}_{ti:04d}", *synth_trajectory(rng))
            n_groups = N_TRAJ
            if fi == 0:
                t, y, P, T_K, seed_attr = synth_trajectory(rng)
                y[1:, ZERO_SPECIES_INDEX] = 0.0
                write_group(f, ZERO_GROUP, t, y, P, T_K, seed_attr)
                t, y, P, T_K, seed_attr = synth_trajectory(rng)
                write_group(f, NAN_GROUP, np.full_like(t, np.nan), np.full_like(y, np.nan), P, T_K, seed_attr)
                n_groups += 2
        print(f"wrote {path} ({n_groups} trajectories)")


if __name__ == "__main__":
    main()
