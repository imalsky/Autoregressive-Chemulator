# Auto-Chem (Autoregressive Chemulator)

Gas-phase chemical kinetics of exoplanet atmospheres for use inside a general circulation model (GCM): one call per cell
and time step advances a cell's composition by dt at its temperature T and pressure P. The code is a physical 0-D solver
of VULCAN's C-H-N-O network (53 species, 290 reversible reactions) as CUDA kernels batched over cells on the GPU, with
derivatives of every call, so machine learning can enter where it helps (per-cell rate multipliers, the Arrhenius
table).

Earlier versions were neural networks that predict the next state directly. Their code is kept at tags:
[`paper-v1`](https://github.com/imalsky/Autoregressive-Chemulator/tree/paper-v1) (the published 12-species model),
`e13s_CBg_s2` (the closed-state model CBg) and `direct-state-final` (the last version of that line: miss and rollout
losses, flow-map and mixed-mode heads).

## The solver

- Mass-action kinetics with forward rates from VULCAN's reaction list and reverse rates from NASA-9 thermodynamics
  (`lib/kinvulcan.py` reads VULCAN-format networks and NASA-9 files as data). Elements, positivity and the equilibrium
  of the thermodynamic data hold by construction.
- Constant T and P within a call: the mole fractions follow dx/dt = (S v - x sum(S v)) / n_tot, with n_tot = P / kT.
- `lib/kincuda.py`: the production call, `CudaSolver(net, device)(x, T, P, dt, h)`, as CUDA kernels generated per
  network and compiled at run time with NVRTC (two or four warps per cell; small batches, up to one cell per SM, run one
  cell per block, with the same results). Per cell: the rate coefficients from T and P, then Rodas3 (order 3, L-stable)
  with the exact Jacobian, the sparse LU without row exchanges in a minimum-degree order (the densest trailing block
  factored in registers), adaptive steps (rtol 1e-3, atol 1e-18 in mole fraction) and the cell's step size carried to
  its next call. A guard kernel flags cells with a non-finite state, an unfinished call or an element drift beyond the
  network's allowance, and a recover kernel redoes them on the GPU (dense LU with threshold pivoting, twice the step
  budget), so a call never waits for the host; the counts are read on demand, and `check()` raises for gas below 300 K.
  Any VULCAN-format network up to 254 species.
- `lib/kincuda_ad.py`: tangent and adjoint kernels of one call (state, T, P, dt, step size, per-cell rate multipliers,
  Arrhenius table), with the step control differentiated (default) or the steps frozen; recovered cells are
  differentiated on the recovery's path (NaN when still failed); no host sync.
- `lib/kinjax.py` (+ `lib/kinjax.cc`, an XLA FFI shim built with g++): the call and its derivatives in JAX (jit, vmap,
  grad, jvp in one tangent call, scan), bitwise the CUDA solvers.

## Layout

- `lib/`: `kinvulcan.py`, `kincuda.py`, `kincuda_ad.py`, `kinjax.py` + `kinjax.cc`; `paths.py` (every data location).
- `bench/kintest.py`: the fast test runner of the GPU solver against VULCAN (quick and full tiers, misses first, paired
  with a saved run).
- `run/env.sh`: environment (GPU, threads, `CHEM_DATA`).
- `tests/kincuda_check.py`: the CUDA kernels against stored outputs of a torch reference solver on the same inputs
  (parity on cold, scrambled and 2,000-call chained calls; guard and recovery); `tests/kincuda_ad_check.py`: the
  derivatives against a stored torch-autograd reference, central differences and each other; `tests/kinjax_check.py`:
  the JAX layer bitwise against the CUDA solvers; `tests/libref.py`: reads the stored reference (each entry checked
  against a digest of its inputs).

## Data and components not included

The VULCAN runs, truth sets, benches and the checks' stored references are not in this repository. They live in the
git-ignored `data/` folder, which `lib/paths.py` locates from one environment variable, `CHEM_DATA` (default: `data/` in
this repository; `run/env.sh` exports it).

Two components are separate and not included, because they are GPL-licensed: the training-data generator (a 0-D wrapper
of VULCAN) and mini-chem. This repository only reads their output files (and VULCAN's reaction and thermodynamic data
files).

## Requirements

Linux with an NVIDIA GPU (the kernels run in fp64; tested on sm_89). Python 3.12.14 (`.python-version`):

    uv venv --python 3.12.14 && uv pip sync requirements.lock

`requirements.lock` pins the environment (torch 2.8.0+cu126 with the CUDA NVRTC and runtime wheels, numpy, JAX 0.12.0.1
with its CUDA 12 plugin; it also holds packages this code no longer imports, such as h5py, scipy, Lightning and
matplotlib). The JAX shim is built with g++ at first use (no nvcc needed). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
when the GPU is shared.

## License and citation

MIT (`LICENSE`). Citation metadata: `CITATION.cff`.
