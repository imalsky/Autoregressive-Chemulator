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
- `lib/kincuda.py`: the same call as CUDA kernels generated per network and compiled at run time with NVRTC (two or
  four warps per cell: rates, every Rodas3 step and the sparse LU without row exchanges in a minimum-degree order, the
  densest trailing block factored in registers); a guard kernel flags cells with a non-finite state, an unfinished call
  or an element drift, and a recover kernel redoes them on the GPU (dense LU with threshold pivoting, twice the step
  budget), so a call never waits for the host. Any VULCAN-format network up to 254 species.
- `lib/kincuda_ad.py`: tangent and adjoint kernels of one call (state, T, P, dt, step size, per-cell rate multipliers,
  Arrhenius table), with the step control differentiated (default) or the steps frozen.
- `lib/kinjax.py` (+ `lib/kinjax.cc`, an XLA FFI shim built with g++): the call and its derivatives in JAX (jit, vmap,
  grad, jvp, scan).

## Layout

- `lib/`: `kinetics.py`, `kinvulcan.py`, `kinbatch.py`, `kinnet.py`, `kincuda.py`, `kincuda_ad.py`, `kinjax.py` +
  `kinjax.cc`; `paths.py` (every data location).
- `bench/`: `score_design.py` (score designs from VULCAN outputs), `score.py` (the averaged score and its comparisons),
  `kintest.py` (the fast test runner of the GPU solver: quick and full tiers, misses first, paired with a saved run).
- `datagen/build_kin.py`: VULCAN clean-design runs mapped onto a reduced species set (rate-network training set).
- `run/env.sh` (environment: GPU, threads, `CHEM_DATA`), `run/runner.sh` (a job lane).
- `tests/kincuda_check.py`: the CUDA kernels against the library path (parity, guard and recovery, paired timing);
  `tests/kincuda_ad_check.py`: the derivatives against a torch-autograd reference, central differences and each other;
  `tests/kinjax_check.py`: the JAX layer bitwise against the CUDA solvers; `tests/score_regress.py`: the scorer on
  hand-made cases.

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

`requirements.lock` also pins JAX 0.12.0.1 with its CUDA 12 plugin; the JAX shim is built with g++ at first use (no
nvcc needed). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when the GPU is shared.

## License and citation

MIT (`LICENSE`). Citation metadata: `CITATION.cff`.
