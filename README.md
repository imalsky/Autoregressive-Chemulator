# Auto-Chem (Autoregressive Chemulator)

Gas-phase chemical kinetics of exoplanet atmospheres for use inside a general circulation model (GCM): one call per cell
and time step advances a cell's composition by dt at its temperature T and pressure P. The current code is a physical
0-D solver of VULCAN's C-H-N-O network (53 species, 290 reversible reactions), batched over cells on the GPU, with
machine learning where it helps (learned rate factors).

Earlier versions were neural networks that predict the next state directly. Their code is kept at tags:
[`paper-v1`](https://github.com/imalsky/Autoregressive-Chemulator/tree/paper-v1) (the published 12-species model),
`e13s_CBg_s2` (the closed-state model CBg) and `direct-state-final` (the last version of that line: miss and rollout
losses, flow-map and mixed-mode heads).

## The solver

- Mass-action kinetics with forward rates from VULCAN's reaction list and reverse rates from NASA-9 thermodynamics
  (`lib/kinvulcan.py` reads VULCAN's network and NASA-9 files as data; `lib/kinetics.py` is the numpy reference with a
  scipy BDF integrator). Elements, positivity and the equilibrium of the thermodynamic data hold by construction.
- Constant T and P within a call: the mole fractions follow dx/dt = (S v - x sum(S v)) / n_tot, with n_tot = P / kT.
- `lib/kinbatch.py`: batched fp64 Rosenbrock steps with the exact Jacobian (ROS2 and Rodas3; Rodas3 without pivoting is
  the production step). `lib/kinnet.py`: adaptive call loops (`advance_compact` steps only the cells still running and
  carries each cell's step size to its next call), learned per-reaction rate factors (`RateNet`, `RateNet2`) and
  gradients through the accepted steps.
- `TorchRates` raises for gas below 300 K.
- Production settings: Rodas3, rtol 1e-3, atol 1e-18 (mole fraction).

## Layout

- `lib/`: `kinetics.py`, `kinvulcan.py`, `kinbatch.py`, `kinnet.py`; `paths.py` (every data location).
- `bench/`: `score_design.py` (score designs from VULCAN outputs) and `score.py` (the averaged score and its comparisons).
- `datagen/build_kin.py`: VULCAN clean-design runs mapped onto a reduced species set (rate-network training set).
- `run/env.sh` (environment: GPU, threads, `CHEM_DATA`), `run/runner.sh` (a job lane).
- `tests/score_regress.py`: the scorer on hand-made cases.

## Data and components not included

The VULCAN runs, truth sets, benches and checkpoints are not in this repository. They live in the git-ignored `data/`
folder, which `lib/paths.py` locates from one environment variable, `CHEM_DATA` (default: `data/` in this repository;
`run/env.sh` exports it).

Two components are separate and not included, because they are GPL-licensed: the training-data generator (a 0-D wrapper
of VULCAN) and mini-chem. This repository only reads their output files (and VULCAN's reaction and thermodynamic data
files).

## Requirements

Linux with a CUDA GPU (the batched solver runs in fp64 under `torch.compile`). Python 3.12.14 (`.python-version`):

    uv venv --python 3.12.14 && uv pip sync requirements.lock

`requirements.lock` pins the environment (torch 2.8.0+cu126, numpy, h5py; it also holds packages this code no longer
imports, such as Lightning and matplotlib).

## License and citation

MIT (`LICENSE`). Citation metadata: `CITATION.cff`.
