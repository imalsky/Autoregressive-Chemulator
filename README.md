# Auto-Chem

Flow-map neural emulators for stiff chemistry trajectories. Given a state `y` at
time `t`, the models predict `y(t + dt)` under variable timesteps:

    y_{t+1} = F(y_t, dt_t, g)

where `g = (P, T)` are per-trajectory global conditioning parameters. See
[`spec.md`](spec.md) for the full design.

## Layout

    config.json           canonical training config (strict schema)
    configs/              alternate config snapshots
    data/                 raw HDF5 + processed NPZ shards live here
    processing/
      preprocessing.py    HDF5 → normalized NPZ shards
      testing.py          sanity checks for preprocessed data
      testing_data.py     sample-data generator
    src/
      main.py             training entrypoint
      dataset.py          shard loader + preloaded batch stream
      model.py            FlowMapMLP / FlowMapAutoencoder
      trainer.py          LightningModule + Trainer factory
      utils.py            shared config + precision helpers
    testing/
      export.py           export to 1-step physical-space .pt2
      aoti_export.py      AOT Inductor packaging + validation
      predictions.py      autoregressive rollout plots
      benchmark.py        throughput benchmarks
      training_logs.py    training curve plots

## Pipeline

### 1. Preprocess raw trajectories

    python -u processing/preprocessing.py

Reads `config.json`, writes `data/processed/{train,validation,test}/shard_*.npz`
and `data/processed/normalization.json`. The `_tmp_physical/` staging dir is
cleaned up automatically on success.

### 2. Train

    python -u src/main.py

Uses the same `config.json`. Override its location with
`AUTOCHEM_CONFIG_PATH=/path/to/cfg.json`. Outputs land under `paths.work_dir`:

    metrics.csv                 per-epoch metrics (CSVLogger)
    config.resolved.json        portable snapshot of the config actually used
    checkpoints/epoch*.ckpt     Lightning checkpoints

Fresh training requires an empty `work_dir`. To continue a run, set
`training.checkpoint_mode = "resume"` and point `runtime.checkpoint` at a
checkpoint file. `"weights_only"` loads weights without optimizer/scheduler
state.

### 3. Export and inspect

    python -u testing/export.py           # writes a 1-step physical-space .pt2
    python -u testing/aoti_export.py      # AOT Inductor package
    python -u testing/predictions.py      # rollout plots against a shard
    python -u testing/benchmark.py        # throughput microbenchmarks

## HPC

`run.pbs` (training) and `testing/run.pbs` (post-processing) are PBS job scripts
targeting Grace-Hopper nodes.

## Design stance

The codebase is intentionally strict: unknown config keys, dtype mismatches,
ambiguous tensor shapes, and missing files are hard errors rather than silent
fallbacks. The rationale is deterministic behavior across environments and easy
reproduction from `config.resolved.json`.
