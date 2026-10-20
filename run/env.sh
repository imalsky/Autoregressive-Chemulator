# Shared environment for this repository's GPU/CPU jobs on daw (sourced by every driver; tmux does not pass the caller's
# environment). H = this repository; data roots as in lib/paths.py (CHEM_PROJECT, CHEM_SCRATCH); E = the E13 bench folder.
export H=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
export PY=$H/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/home/imalsky/tmp
export TORCHINDUCTOR_CACHE_DIR=/home/imalsky/tmp/sweep_inductor
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # ordinal 1 = B5:00.0 (nvidia-smi index 1, NUMA node 1)
export CHEM_PROJECT=${CHEM_PROJECT:-/home/imalsky/Chemulator_Project} CHEM_SCRATCH=${CHEM_SCRATCH:-/scratch/imalsky}
export E=$CHEM_PROJECT/experiments/2026-10-03_repo-sweep/_build/g2/E13_gcm
