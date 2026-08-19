#!/usr/bin/env python3
"""make_synthetic_raw.py — tiny synthetic raw HDF5 files in the VULCAN schema.

Purpose: local end-to-end smoke testing of preprocessing + training configs
(see testing/run_local_smoke.sh) on machines that do not hold the
real raw dataset. NOT for science: species curves are smooth synthetic decays.

Schema produced (matches processing/preprocessing.py expectations):
  file.h5
    run_NNNN/                  one group per trajectory
      t_time                   [T] float64, strictly increasing
      <species>_evolution      [T] float64, finite, > drop_below
      attrs: P, T              finite scalars (physical units)

Usage:
    AUTOCHEM_SYNTH_OUT_DIR=/tmp/ac_smoke_raw python processing/make_synthetic_raw.py
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

T_POINTS = 600
T_START, T_END = 1e-6, 5e5   # spans the full production dt band x n_steps

# Generation settings (no argparse; module-level constants per spec 9.3.4).
# The output directory is the one run-varying input and comes from a required
# environment variable (mirrors the AUTOCHEM_CONFIG_PATH fail-fast convention).
OUT_DIR_ENV_VAR = "AUTOCHEM_SYNTH_OUT_DIR"
N_FILES = 2
N_TRAJ = 30  # trajectories per file
SEED = 0


def synth_trajectory(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, float]:
    t = np.logspace(np.log10(T_START), np.log10(T_END), T_POINTS)
    s = len(SPECIES)
    # Smooth positive curves: equilibrium level + decaying component, in log10 space.
    log_eq = rng.uniform(-20.0, -0.5, size=s)
    log_init = log_eq + rng.uniform(-3.0, 3.0, size=s)
    tau = 10.0 ** rng.uniform(-2.0, 4.0, size=s)
    decay = np.exp(-t[:, None] / tau[None, :])
    log_y = log_eq[None, :] + (log_init - log_eq)[None, :] * decay
    y = 10.0 ** log_y
    P = float(10.0 ** rng.uniform(-2.0, 8.0))   # Pa
    T_K = float(rng.uniform(300.0, 3000.0))
    return t, y, P, T_K


def _resolve_out_dir() -> Path:
    raw = os.environ.get(OUT_DIR_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{OUT_DIR_ENV_VAR} is not set. Point it at the output directory, e.g.:\n"
            f"  {OUT_DIR_ENV_VAR}=/tmp/ac_smoke_raw python processing/make_synthetic_raw.py"
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
                grp = f.create_group(f"run_{fi:03d}_{ti:04d}")
                t, y, P, T_K = synth_trajectory(rng)
                grp.create_dataset("t_time", data=t)
                for j, name in enumerate(SPECIES):
                    grp.create_dataset(name, data=y[:, j])
                grp.attrs["P"] = P
                grp.attrs["T"] = T_K
        print(f"wrote {path} ({N_TRAJ} trajectories)")


if __name__ == "__main__":
    main()
