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
    src/
      main.py             training entrypoint
      dataset.py          shard loader + preloaded batch stream
      model.py            FlowMapMLP (the flow-map network)
      trainer.py          LightningModule + Trainer factory
      ema_callback.py     EMA-of-weights Lightning callback
      utils.py            shared config + precision helpers
    testing/
      export.py           export to 1-step physical-space .pt2 (deployment artifact)
      aoti_export.py      AOT Inductor packaging + validation
      predictions.py      autoregressive rollout plots (diagnostic)
      training_logs.py    training-curve plots (diagnostic)
      eval_rollout_error.py   percentile rollout-error vs step (paper headline metric)
      bench_speedup.py    exported-model (CPU and CUDA) latency sweep vs mini-chem per-cell cost
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
`data/processed/normalization.json`. A rebuild leaves the previous dataset in place
until the new one is complete (it is built in `_tmp_final/` and swapped in, with
`normalization.json` last). The scratch dirs (`_tmp_physical/`, `_tmp_final/`,
`_tmp_old/`) are removed on success; after a failure the next run wipes them
(`preprocessing.overwrite=true`) or refuses (`false`). Stage 2 reuses these same shards.
`normalization.json` also records the input units and the trained domain (window start
times, (T, P) coverage, the tracked-species sum); `preprocessing_summary.json` is the
build record (per raw file: size, groups, rejects, windows; the config blocks used; the
normalization fingerprint). Training refuses a processed dir whose record does not match
its config (except `overwrite`, `num_workers`), whose `normalization.json` was changed
after the build, or (fresh runs) whose raw files were added or resized since: rebuild
it, into a new `processed_dir` if the old one must stay.

### 2. Train stage 1 (one-jump pretrain)

    AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u src/main.py

Outputs land under `paths.work_dir` (`models/stage1`):

    metrics.csv                 per-epoch metrics (CSVLogger)
    train.log                   run log (appended on resume)
    config.resolved.json        portable snapshot of the config the run was started with
    config.resume[.N].json      the config of each resume (must match config.resolved.json)
    provenance.jsonl            one line per invocation: code, environment, data and init-checkpoint identity
    checkpoints/epoch*.ckpt     top-k Lightning checkpoints by val_loss (EMA weights embedded)
    checkpoints/last.ckpt       the latest completed epoch
    checkpoints/best.ckpt       stable-name copy of the best-val_loss checkpoint (finished runs only)
    COMPLETED                   written last, only by a finished run (max_epochs or early stopping)

An epoch is `dataset.train_batches_per_epoch` random batches; validation scores the
fixed set of `dataset.val_samples` windows (see docs/DECISIONS.md). Every checkpoint
records the normalization fingerprint (`norm_sha256`) of the processed data it was
trained on.

### 3. Train stage 2 (autoregressive fine-tune)

    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u src/main.py

`stage2.json` loads stage-1 `checkpoints/best.ckpt` via `checkpoint_mode="weights_only"`
(EMA weights are preferred automatically) and fine-tunes with a rollout
curriculum. The stage-1 run must have finished (`COMPLETED`) on the same processed
data (matching `norm_sha256`). Before training, the loaded weights are scored once on
the validation set (`init_val.json`). Outputs land under `models/stage2`.

Fresh training requires an empty `work_dir` (one holding only `train.log` from a
failed start counts as empty). Only one training process may use a `work_dir` (it
is flocked). To continue a run in place, set `training.checkpoint_mode="resume"` and
point `runtime.checkpoint` at a checkpoint in that run's own `checkpoints/` (normally
`last.ckpt`); the config must equal `config.resolved.json` except
`training.checkpoint_mode`, `runtime.checkpoint` and `training.max_epochs`. A moved or
copied run directory cannot be resumed: start a new run from its weights with
`checkpoint_mode="weights_only"`. A finished run (`COMPLETED`, or a checkpoint from its
last epoch or after early stopping) is not resumed: to extend it deliberately, delete
`COMPLETED` and raise `training.max_epochs`. (The HPC scripts resume automatically on requeue
once at least one checkpoint exists; a work dir from a run that died before its
first checkpoint must be removed by hand.)

### 4. Export and inspect

    python -u testing/export.py           # 1-step physical-space .pt2 (edit RUN_DIR; defaults to models/stage2)
    python -u testing/aoti_export.py      # AOT Inductor package + validation
    python -u testing/predictions.py      # rollout plots against a shard
    python -u testing/bench_speedup.py    # exported-model latency sweep (edit RUN_DIR)

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
