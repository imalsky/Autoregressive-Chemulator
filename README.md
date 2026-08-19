# Auto-Chem

Flow-map neural emulators for stiff chemistry trajectories. Given a state `y` at
time `t`, the model predicts `y(t + dt)` under variable timesteps:

    y_{t+1} = F(y_t, dt_t, g)

where `g = (P, T)` are per-trajectory global conditioning parameters. The
production model is a single dt-conditioned MLP flow-map trained in two stages
(one-jump pretrain -> autoregressive fine-tune). See [`spec.md`](spec.md) for the
full design and [`docs/DECISIONS.md`](docs/DECISIONS.md) for the evidence behind
every non-default config value.

## Layout

    configs/
      stage1.json         production stage 1 (one-jump pretrain) config
      stage2.json         production stage 2 (autoregressive fine-tune) config
    docs/
      DECISIONS.md        per-key evidence + justification for the configs
      AR_PAPER_PLAN.md    ApJ paper plan, outline, and evidence map
    data/                 raw HDF5 + processed NPZ shards live here (not source)
    models/               training outputs / checkpoints live here (not source)
    processing/
      preprocessing.py    HDF5 -> normalized NPZ shards (canonical)
      make_synthetic_raw.py   tiny synthetic raw generator (for the smoke test)
      testing.py          raw-vs-chunk overlay plots (diagnostic)
      testing_data.py     raw HDF5 structure scan (diagnostic)
    src/
      main.py             training entrypoint
      dataset.py          shard loader + preloaded batch stream
      model.py            FlowMapMLP / FlowMapAutoencoder
      trainer.py          LightningModule + Trainer factory
      ema_callback.py     EMA-of-weights Lightning callback
      utils.py            shared config + precision helpers
    testing/
      export.py           export to 1-step physical-space .pt2 (deployment artifact)
      aoti_export.py      AOT Inductor packaging + validation
      predictions.py      autoregressive rollout plots (diagnostic)
      benchmark.py        throughput benchmarks (diagnostic)
      training_logs.py    training-curve plots (diagnostic)
      eval_rollout_error.py   percentile rollout-error vs step (paper headline metric)
      bench_speedup.py    exported-model latency sweep vs mini-chem per-cell cost
      bench_eager_devices.py  eager CPU/MPS latency (GPU half of the speed benchmark)
      science.mplstyle    shared matplotlib style for the paper figures
      run.pbs             HPC: post-training export + plots job
      run_local_smoke.sh  end-to-end smoke on synthetic data
    run.pbs               HPC: stage 1 (preprocess + pretrain)
    run_stage2.pbs        HPC: stage 2 (autoregressive fine-tune)

There is **no default config**. Every entrypoint requires the config to be named
explicitly via the `AUTOCHEM_CONFIG_PATH` environment variable; running without it
is a hard error. Config paths (`paths.*`, `runtime.checkpoint`) are resolved
relative to the config file's directory.

## Pipeline

### 1. Preprocess raw trajectories (once)

    AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u processing/preprocessing.py

Writes `data/processed/{train,validation,test}/shard_*.npz` and
`data/processed/normalization.json`. The `_tmp_physical/` staging dir is cleaned
up automatically on success. Stage 2 reuses these same shards.

### 2. Train stage 1 (one-jump pretrain)

    AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u src/main.py

Outputs land under `paths.work_dir` (`models/stage1`):

    metrics.csv                 per-epoch metrics (CSVLogger)
    config.resolved.json        portable snapshot of the config actually used
    checkpoints/epoch*.ckpt     Lightning checkpoints (EMA weights embedded)
    checkpoints/best.ckpt       stable-name copy of the best-val_loss checkpoint

### 3. Train stage 2 (autoregressive fine-tune)

    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u src/main.py

`stage2.json` loads the stage-1 checkpoint via `checkpoint_mode="weights_only"`
(EMA weights are preferred automatically) and fine-tunes with a rollout
curriculum. Outputs land under `models/stage2`.

Fresh training requires an empty `work_dir`. To continue a run in place, set
`training.checkpoint_mode="resume"` and point `runtime.checkpoint` at a
checkpoint file (the HPC scripts do this automatically on requeue once at least
one checkpoint exists; a work dir from a run that died before its first
checkpoint must be removed by hand).

### 4. Export and inspect

    python -u testing/export.py           # 1-step physical-space .pt2 (edit RUN_DIR; defaults to models/stage2)
    python -u testing/aoti_export.py      # AOT Inductor package + validation
    python -u testing/predictions.py      # rollout plots against a shard
    python -u testing/benchmark.py        # throughput microbenchmarks

A fast end-to-end check on synthetic data (no real dataset needed):

    conda activate nn && bash testing/run_local_smoke.sh

## HPC

`run.pbs` runs stage 1 (preprocess-if-needed + pretrain); `run_stage2.pbs` runs
stage 2. Both target Grace-Hopper (GH200) nodes, auto-resume on requeue once at
least one checkpoint exists, and read `configs/stage{1,2}.json`. Typical
submission:

    qsub run.pbs
    qsub -W depend=afterok:<stage1_jobid> run_stage2.pbs

## Design stance

The codebase is intentionally strict: unknown config keys, dtype mismatches,
ambiguous tensor shapes, missing files, and a missing `AUTOCHEM_CONFIG_PATH` are
hard errors rather than silent fallbacks. Required config keys are read directly
(no silent defaults); only genuinely optional feature blocks (`model.fourier_dt`,
`training.autoregressive_training.loss_discount_gamma`) carry documented,
named-constant defaults. The rationale is deterministic behavior across
environments and easy reproduction from `config.resolved.json`.
