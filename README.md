# Auto-Chem

Flow-map neural emulators for stiff chemistry trajectories. Given a state `y` at
time `t`, the model predicts `y(t + dt)` for a timestep `dt` in the trained range:

    y_{t+1} = F(y_t, dt_t, g)

The one-step export accepts any in-range dt per call and per row; multi-step rollouts are trained
and validated only at constant dt within a rollout; rollouts whose dt changes between steps are
unvalidated. Here `g = (P, T)` are per-trajectory global conditioning parameters. The
production model is a single dt-conditioned MLP flow-map trained in two stages
(one-jump pretrain -> autoregressive fine-tune). See [`spec.md`](spec.md) for the
full design and [`docs/DECISIONS.md`](docs/DECISIONS.md) for the per-key evidence behind
the config values (values without evidence are listed there as inherited / unjustified).

## Layout

    configs/
      stage1.json         production stage 1 (one-jump pretrain) config
      stage2.json         production stage 2 (autoregressive fine-tune) config
    docs/
      DECISIONS.md        per-key evidence for the configs + inherited / unjustified values
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
      eval_rollout_error.py   rollout-error curves vs step over a seeded validation sample (per-species dex),
                              with persistence, loss vs persistence by window-start decade and gate-1 lines
                              on the evolving windows (the headline), steady-state windows reported apart
                              by absolute log10 error
      bench_speedup.py    exported-model (CPU and CUDA) latency sweep vs mini-chem per-cell cost
      run_local_smoke.sh  end-to-end smoke on synthetic data (the integration gate)
    run_daw.sh            daw launcher: preprocess -> stage 1 -> stage 2, each with its post step
    requirements.lock     exact package pins (the daw venv); .python-version: 3.12.14
    LICENSE               MIT license
    CITATION.cff          software citation metadata

## Environment

The project runs on daw from a uv venv in the repo root, installed from the lock file:

    uv venv --python 3.12.14 && uv pip sync requirements.lock

`python` in the commands below is `.venv/bin/python` (or activate `.venv`). For training,
use `run_daw.sh` (below): it also pins the GPU, the CPUs and the thread counts.

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
(`preprocessing.overwrite=true`) or refuses (`false`). One preprocessing run at a time per
processed dir: it holds a lock on `data/processed.lock`, a persistent 0-byte file next to the
processed dir, and a second run (or leftover workers of a killed one) exits with an error.
Training holds the same lock shared from reading `normalization.json` until both splits are
preloaded (`eval_rollout_error.py` while it loads its rows), so neither waits: a training start
fails while preprocessing runs, and preprocessing fails while a training start is loading. Stage 2
reuses these same shards.
A trajectory whose raw output froze in mid-evolution (the generator repeated its last state
while a species was still changing by more than 1e-3 dex per raw interval) is cut after its last
changing row, so its windows come only from the evolving part; one with no room left for a
window is rejected (`frozen_tail_no_window`). Converged repeated tails are kept. On the 4 raw
files this cuts 16,311 of 649,084 kept trajectories and rejects 986 of them (docs/DECISIONS.md,
open item 17).
`normalization.json` also records the input units and the trained domain (window start
times, (T, P) coverage, the tracked-species sum); `preprocessing_summary.json` is the
build record (per raw file: size, groups, rejects, windows; the frozen-tail counts; the config
blocks used; the normalization fingerprint). Training refuses a processed dir whose record does
not match its config (except `overwrite`, `num_workers`), that was built before the frozen-tail
cut (2026-10-01), whose `normalization.json` was changed
after the build, or (fresh runs) whose raw files were added or resized since: rebuild
it, into a new `processed_dir` if the old one must stay.

### 2. Train stage 1 (one-jump pretrain)

    AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u src/main.py

Outputs land under `paths.work_dir` (`models/stage1`):

    metrics.csv                 per-epoch metrics (CSVLogger), incl. timing/throughput (epoch_time_sec, perf_*; src/trainer.py)
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
trained on. A non-finite val_loss, or a non-finite training loss anywhere in the epoch
(checked at the end of the epoch, not at the step), aborts the run (exit 1) before that
epoch's checkpoints are written.

### 3. Train stage 2 (autoregressive fine-tune)

    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u src/main.py

`stage2.json` loads stage-1 `checkpoints/best.ckpt` via `checkpoint_mode="weights_only"`
(EMA weights are preferred automatically) and fine-tunes with a rollout
curriculum. The stage-1 run must have finished (`COMPLETED`) on the same processed
data (matching `norm_sha256`), and its `best.ckpt` must have the sha256 that `COMPLETED`
records (any other checkpoint of that run loads with a warning naming the recorded best). Its peak LR is min(`training.optimizer.lr`, the LR saved
in the loaded checkpoint), logged and recorded in `provenance.jsonl` (`handoff_peak_lr`);
a resume keeps it. Before training, the loaded weights are scored once on the validation
set (`init_val.json`, with `val_persistence_loss` and `val_skill`). Outputs land under
`models/stage2`.

Fresh training requires an empty `work_dir` (one holding only `train.log` from a
failed start counts as empty). Only one training process may use a `work_dir` (it
is flocked). To continue a run in place, set `training.checkpoint_mode="resume"` and
point `runtime.checkpoint` at a checkpoint in that run's own `checkpoints/` (normally
`last.ckpt`); the config must equal `config.resolved.json` except
`training.checkpoint_mode`, `runtime.checkpoint` and `training.max_epochs`. A moved or
copied run directory cannot be resumed: start a new run from its weights with
`checkpoint_mode="weights_only"`. A finished run (`COMPLETED`) is not resumed: to extend it
deliberately, delete `COMPLETED` and raise `training.max_epochs`. A resume from a run's final
checkpoint (its last epoch, or after early stopping) in a work dir without `COMPLETED`, i.e. a
run that died after its last `last.ckpt` but before publishing, publishes `best.ckpt` and
`COMPLETED` from that checkpoint without training and exits 0. (`run_daw.sh` resumes an
interrupted stage automatically once its `last.ckpt` exists; a work dir from a run that died
before its first checkpoint must be removed by hand.)

### 4. Export and inspect

Every tool acts on the run named by `AUTOCHEM_CONFIG_PATH` (its `paths.work_dir`), so the
same commands serve stage 1 and stage 2:

    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/export.py              # 1-step physical-space .pt2 (CPU + CUDA)
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/aoti_export.py         # AOT Inductor package + validation (not on daw)
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/eval_rollout_error.py  # rollout-error curves (validation split)
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/predictions.py         # rollout plot against a shard
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/bench_speedup.py       # exported-model latency sweep
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/training_logs.py       # loss curves
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/efficiency_summary.py  # throughput, TFLOP/s, GPU logger summary
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/profile_training.py    # torch.profiler summary + trace of a short fit
                                                                                      # (writes <work_dir>_profile/; never the run itself)

Export requires a finished run (`COMPLETED`) whose checkpoint matches its processed data
(`norm_sha256`); every requested device must be available. Inputs are in physical units
(P in dyn/cm^2, T in K, dt in s, species mixing ratios relative to the whole gas; see spec 4.3);
a row whose dt, P or T lies outside the trained range, or whose inputs are not finite, comes back
as NaN. The training compositions contain no helium, so helium-bearing compositions are outside the
validated domain; how a helium-bearing host should map its composition is an open decision
(docs/DECISIONS.md, open item 25). Each artifact is checked against the eager model before it
replaces the previous one.
`aoti_export.py` installs its package only when it matches the fresh export (otherwise it
writes its reports and exits 1); the package does no input validation, so hosts must call
it through `_check_step_inputs` or apply the same checks (spec 4.3). It is not run on daw
(decided 2026-10-01): on torch < 2.13, loading an AOTI `.pt2` writes about 152 MB per load (at
production size) into a directory under the root temp dir whatever `TMPDIR` says, and a killed run
leaves it there. The torch.export `.pt2` is the deployment artifact; AOTI is revisited with a
project-wide upgrade to torch >= 2.13 on its own branch, gated on parity (docs/DECISIONS.md, open
item 26).

The integration gate, about two minutes on one GPU with synthetic data (no real dataset needed):

    bash run_daw.sh smoke

It runs the production configs, shrunk, through preprocessing, both stages with compile on, a
deliberate extension and a refused re-run of the finished stage 1, both stage-2 recipes, the
Fourier-dt fallback pair and the gated export, and fails on the first wrong outcome (header of
`testing/run_local_smoke.sh`). Run it before committing changes to `src/`, `processing/` or the
export.

## Running on daw

    tmux new -s autochem 'bash run_daw.sh all'     # or: preprocess | stage1 | stage2 | smoke

`run_daw.sh` preprocesses once (skipped while `data/processed/normalization.json` exists),
trains stage 1 and then stage 2 from stage 1's `best.ckpt`, and after each stage runs its post
step (`export.py`, `training_logs.py`, `predictions.py`, `eval_rollout_error.py`,
`efficiency_summary.py`). Re-running it is safe: a finished stage (`COMPLETED`) is not
retrained, an interrupted one resumes from `last.ckpt` (one that had finished training but not
published is published from it), and refusals from `src/main.py` are reported, not retried. It takes one GPU (the first idle one, or `AUTOCHEM_GPU`; it refuses a
second GPU for the same user), pins itself to 16 CPUs on that GPU's NUMA node
(`AUTOCHEM_THREADS`; all child processes inherit the pin), sets the thread counts, `TMPDIR`
and the compile caches (inductor compiles in-process, `TORCHINDUCTOR_COMPILE_THREADS=1`, so no
compile-worker pool puts semaphores on the shared RAM disk), and stops cleanly on Ctrl-C or
`kill <pid>` (training exits 143 and resumes on the next run). Console logs, the GPU logger CSV and the last resume config go to
`logs/`.

## Design stance

The codebase is intentionally strict: unknown config keys, dtype mismatches,
ambiguous tensor shapes, missing files, and a missing `AUTOCHEM_CONFIG_PATH` are
hard errors rather than silent fallbacks. Required config keys are read directly
(no silent defaults); only genuinely optional feature blocks (`model.fourier_dt`,
`training.autoregressive_training.loss_discount_gamma`) carry documented,
named-constant defaults. The rationale is deterministic behavior across
environments and easy reproduction from `config.resolved.json`.

## License

MIT (see `LICENSE`). Citation metadata for the software is in `CITATION.cff`.
