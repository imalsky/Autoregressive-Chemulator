#!/usr/bin/env python3
"""make_synthetic_raw.py — tiny synthetic raw HDF5 files in the VULCAN schema.

Purpose: local end-to-end smoke testing of preprocessing + training configs
(see configs/production/run_local_smoke.sh) on machines that do not hold the
real raw dataset. NOT for science: species curves are smooth synthetic decays.

Schema produced (matches processing/preprocessing.py expectations):
  file.h5
    run_NNNN/                  one group per trajectory
      t_time                   [T] float64, strictly increasing
      <species>_evolution      [T] float64, finite, > drop_below
      attrs: P, T              finite scalars (physical units)

Usage:
    python processing/make_synthetic_raw.py --out-dir /tmp/ac_smoke_raw \
        --n-files 2 --n-traj 30 --seed 0
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-files", type=int, default=2)
    ap.add_argument("--n-traj", type=int, default=30, help="trajectories per file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for fi in range(args.n_files):
        path = out / f"synthetic_run{fi:03d}.h5"
        with h5py.File(path, "w") as f:
            for ti in range(args.n_traj):
                grp = f.create_group(f"run_{fi:03d}_{ti:04d}")
                t, y, P, T_K = synth_trajectory(rng)
                grp.create_dataset("t_time", data=t)
                for j, name in enumerate(SPECIES):
                    grp.create_dataset(name, data=y[:, j])
                grp.attrs["P"] = P
                grp.attrs["T"] = T_K
        print(f"wrote {path} ({args.n_traj} trajectories)")


if __name__ == "__main__":
    main()
