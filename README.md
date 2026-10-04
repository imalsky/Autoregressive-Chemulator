# Auto-Chem

A neural emulator for the gas-phase chemical kinetics of exoplanet atmospheres. Given the mixing ratios y of 12
species at pressure P and temperature T, the model predicts how they change over a time step dt, so a host model can
call it once per cell and step. It is trained on 0-D VULCAN-0D chemistry runs.

The model is a relaxation head on a 10x2048 residual MLP. In standardized log10 space z, each species relaxes toward a
target, z' = z + (1 - exp(-k dt)) (q - z), where the network predicts the target q and the rate k from (z, P, T)
only. It never sees dt, and q is clamped to 1e-30 <= y <= 1, so a step over zero time changes nothing and the state
stays inside that range. Training has two stages: one-step training, then a long-unroll stage (81 model steps
without gradients, then backpropagation through up to 9 more) with a long-jump loss on single steps of 0.1 s to
1e7 s.

Units: P in dyn/cm^2 (barye), T in K, dt in s, and species (C2H2, CH4, CO2, CO, H2O, H2, HCN, H, N2, NH3, OH, O) as
mixing ratios over the whole gas.

Scope of the training data: VULCAN-0D boxes at fixed T (300-3000 K) and P (1-1e10 dyn/cm^2) with its C-H-O-N
thermochemical network (52 C-H-O-N species), with no photochemistry, no condensation, no sulfur, metal or ion
chemistry, and no helium. Each trajectory starts with H2 as its largest species and the other 11 tracked species
drawn log-uniformly over 20 or 25 dex below it, so solar-like element ratios are rare (C/H, O/H and N/H all within
0.5 dex of solar in about 0.5% of the kept trajectories). The model covers the 12 tracked species only: no heating
rate and no mean molecular weight. The four raw files (224,000 trajectories each) are not in this repository.

## Layout

    configs/       stage1.json, stage2.json: the production configs
    processing/    preprocessing.py (raw HDF5 -> normalized shards and the raw-trajectory store),
                   make_synthetic_raw.py (smoke-test data), testing.py (window figure)
    src/           main.py (training entry point), model.py, trainer.py, dataset.py, ema_callback.py, utils.py
    testing/       export.py, evaluation, plots, benchmarks and the smoke test (run_local_smoke.sh)
    run_daw.sh     launcher: preprocess | stage1 | stage2 | all | smoke
    data/, models/ raw and processed data, run outputs (not in git)

## Install

Linux with a CUDA GPU. From the repository root:

    uv venv --python 3.12.14 && uv pip sync requirements.lock

This installs the exact pins (torch 2.8.0+cu126, Lightning 2.6.6). Every entry point needs `AUTOCHEM_CONFIG_PATH`;
there is no default config.

## Run

    AUTOCHEM_CONFIG_PATH=configs/stage1.json .venv/bin/python -u processing/preprocessing.py   # once
    AUTOCHEM_CONFIG_PATH=configs/stage1.json .venv/bin/python -u src/main.py                   # stage 1
    AUTOCHEM_CONFIG_PATH=configs/stage2.json .venv/bin/python -u src/main.py                   # stage 2
    AUTOCHEM_CONFIG_PATH=configs/stage2.json .venv/bin/python -u testing/export.py             # export

Preprocessing reads every `*.h5` / `*.hdf5` file in `data/raw/` and writes `data/processed/`: shards,
`normalization.json`, `preprocessing_summary.json` and `raw_trajectories/`. Stage 2 starts from a finished stage 1 on
the same processed data and needs `raw_trajectories/` for the long-jump loss; a processed dir built without it must be
moved aside and rebuilt. Runs write to `models/stage1` and `models/stage2`.

The launcher runs the whole pipeline on one GPU with pinned CPUs, resumes an interrupted stage, and exports and
evaluates after each stage. It skips preprocessing while `data/processed/normalization.json` exists:

    tmux new -s autochem 'bash run_daw.sh all'     # or: preprocess | stage1 | stage2 | smoke

Overrides: `AUTOCHEM_GPU`, `AUTOCHEM_THREADS`, `AUTOCHEM_GPU_LOG_S`, `PY`. The integration test, `bash run_daw.sh
smoke`, runs the production configs, shrunk, on synthetic data in a minute or two on one GPU.

## Host interface

The export writes `export_cpu_dynB_1step_dlog10.pt2` and `export_cuda_dynB_1step_dlog10.pt2` (each with a `.sha256`)
from the final epoch of the run: one-step torch.export programs with the normalization baked in. They take float32
`y [B, 12]`, `dt_seconds [B]` and `g [B, 2]` = (P, T), for B up to 16384, and return `dlog10_y [B, 12]`, the change
of log10 y over dt for each row; the host applies it to its own state in its own precision, and `host_step` in
`testing/export.py` is the reference update. dt is per row and per call, trained for 0.1-1000 s; a row whose dt, P or
T lies outside the trained box, or whose inputs are not finite, comes back as NaN (no update).

    import json, torch
    extra = {"metadata.json": ""}
    ep = torch.export.load("export_cuda_dynB_1step_dlog10.pt2", extra_files=extra)
    meta = json.loads(extra["metadata.json"])        # species order, units, trained box, update rule
    with torch.inference_mode():                     # never call .eval() or .train() on ep.module()
        dlog10_y = ep.module()(y, dt_seconds, g)

## License and citation

MIT (see `LICENSE`). Citation metadata is in `CITATION.cff`.
