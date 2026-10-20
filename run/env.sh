# Shared environment for the harness's GPU/CPU jobs (sourced by every job; tmux does not pass the caller's environment).
# 2026-10-09: the sweep's env.sh plus the 10-08 sprint's additions (was env_sprint.sh); the old locations
# (experiments/2026-10-03_repo-sweep/_build/env.sh, experiments/2026-10-08_e13-sprint/_build/env_sprint.sh) link here.
# Data roots as in lib/paths.py: B = sweep _build (runs, vulcan, snapshot), E = E13 benches, R = runs, X = sprint _build.
export PY=/home/imalsky/Chemulator_Project/code/Auto-Chem/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/home/imalsky/tmp
export TORCHINDUCTOR_CACHE_DIR=/home/imalsky/tmp/sweep_inductor
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # ordinal 1 = B5:00.0 (nvidia-smi index 1, NUMA node 1)
export H=/home/imalsky/Chemulator_Project/code/e13-harness
export B=/home/imalsky/Chemulator_Project/experiments/2026-10-03_repo-sweep/_build
export X=/home/imalsky/Chemulator_Project/experiments/2026-10-08_e13-sprint/_build
export E=$B/g2/E13_gcm R=$B/runs
export C17=/scratch/imalsky/arch_gcm17_all C17C=/scratch/imalsky/arch_gcm17cold_all
