# Production config decisions — evidence and justification

## REVISION 2026-10-01: stage-2 recipe R, the handoff LR cap, persistence skill and a provisional gate 1 (author decisions, theme D1)

Author decisions of 2026-10-01 (open items 1, 2, 8, 9 and 10). The persistence diagnostics change
no training numerics, val_loss or monitor (30-step fingerprint bitwise identical to the previous
reference; metrics.csv gains two columns); recipe R and the LR cap change stage 2's schedule.
What changed:
- Stage-2 recipe R (`configs/stage2.json`; values from the audit's recipe analysis, structure-
  derived, untested): 20,000-step epochs (was 93,367), a fixed validation set of 2,621,440 windows
  (was 23,708,768), LR warmup 1 epoch (was 5), curriculum ramp 16 epochs (was 30), plateau
  patience 5 (was 10), early-stopping patience 12 (was 20), max_epochs 120 (was 200); every other
  value as shipped. In steps: warmup 20,000, ramp 320,000, plateau patience 100,000, ES patience
  240,000, cap 2,400,000. Checked with the real classes on CPU (scripted val_loss): both monitors
  first count epoch 16; in a stall the LR is cut at epochs 22 and 28 and early stopping stops at
  28; under slow steady progress the 11th cut, to the 1.46e-8 floor, lands at epoch 82, so a full
  anneal needs at least 83 epochs (10 cuts and 77 epochs from a capped peak of 1.875e-5).
  Projected (estimates from the recipe analysis): about 10 h to 2.3 days fp32 on daw, against
  about 3.5 to 17.6 days for the shipped stage 2. Stage 1 is unchanged.
- Peak LR at a weights_only start: min(`training.optimizer.lr`, the source checkpoint's optimizer
  param-group LR, saved after that epoch's plateau step). It is logged, recorded in
  provenance.jsonl (`handoff_peak_lr`; `init_checkpoint` also records the scheduler's
  `base_lrs`), and restored with the scheduler state on resume. A source checkpoint without that
  optimizer state, or whose param groups disagree, now fails at startup.
- Early stopping keeps the T1 rule (both monitors start counting at the first epoch after the
  warmup and the ramp); option A (keep the warmup-era best) is rejected.
- Validation logs `val_persistence_loss` (the val_loss definition, same weights, mask and windows,
  scored on y_next = y) and `val_skill` = 1 − val_loss / val_persistence_loss from the two epoch
  means, in metrics.csv and in a weights_only run's init_val.json. Diagnostics only: nothing
  monitors them.
- `testing/eval_rollout_error.py` adds persistence curves, a model-vs-persistence loss ratio and
  skill table by window-start decade (the run's val_loss definition recomputed from the export's
  outputs), and the gate-1 pass/fail lines; its headline rows state their step and dt. The shards
  hold no time, so the script now replays each window's start time from the build's raw files
  with preprocessing's own draws and checks every replayed window against its shard row (about
  1.5 min of extra reads on the 4-file build, an estimate from one file's sample).
- Gate 1 is restated as skill against persistence, PROVISIONAL ("Validation protocol"); the
  stage-2 target stays a reported per-species p90 in dex at a stated step and dt.

## REVISION 2026-10-01: documents aligned with the code and the audit's measurements (audit theme T8)

Documentation, two docstrings and two code comments, and `_comment` keys in the configs: no code
path, config value or output changes (`_comment` keys are stripped before every config comparison;
the 30-step fingerprint of both stages is bitwise identical to the previous reference). What
changed:
- False or stale statements are corrected in place and marked "(corrected 2026-10-01)" where they
  sit in a dated block: the production parameter count (37.9M, not ~100M), the data-scale and GPU
  memory notes, the v3 and v2_bigger_batch records behind the batch-size, stage-1 template and
  gate-1 rows (now quoted from git history), the dt-range margin, the List et al. numbers (arXiv
  v2), the E4 and gamma wording (local evidence mixed, not a tie), the banded-layout "fallback",
  the Fourier-dt "scale reversal" (a trunk interaction measured at one scale), and the stage-2
  spot-check (its first 10 epochs cannot separate the two recipes).
- The confidence rubric names the labels the table actually uses; rows whose label the rubric does
  not support carry a bracketed "[rubric: ...]" note, and rows without local evidence get a
  candidate spot-check or a waiver, pending the author.
- Documented alternative 3 (re-preprocess with a narrowed dt range, then fine-tune) is deleted: it
  was not an approved fallback and would re-map the dt input.
- New sections: the evidence locations; the sibling repo's AR recipe and how Auto-Chem differs
  from it; element closure of the 12 tracked species; label accuracy (the interpolation floor); a
  dated re-verification record; and **Open items for the author**, which lists every audit
  decision left to the author with its numbers.
- Every new experiment design, gate or target in this file is a recommendation pending the
  author, not a decision.

## REVISION 2026-10-01: leftover dead paths removed (audit theme T9)

Training, preprocessing and export results are unchanged. Unpatched short fits (no Trainer step
limits; compile off) give the same LR at every step and bitwise-identical weights. With compile on
(CUDA graphs in stage 1), a training fingerprint whose harness overrides make the old and new warmup
formulas agree is bitwise identical. Preprocessing outputs are equal array by array. The smoke
reproduces every printed loss. What changed:
- `runtime.torch_compile.compile_open_loop_unroll` is removed. Validation always runs the compiled
  forward step (per-key table). A config that sets the key fails as an unknown key. Resuming an older
  run is refused until the key is deleted from that run's `config.resolved.json`. No compile-time or
  startup saving is claimed (none measured).
- `model.activation` accepts only `"silu"`, the activation of the production trunk and of both
  documented fallbacks. relu, gelu, tanh, elu and leaky_relu (with its unexplained 0.1 slope) are
  gone; each is one line to restore if the deferred architecture study sweeps activations.
- The LR warmup is read from the config: min(warmup_epochs, max_epochs) ×
  `dataset.train_batches_per_epoch` optimizer steps. This is the value Lightning's estimate gave in
  every production run (no step or batch limits on the Trainer, one process, no accumulation). A
  tool that caps the Trainer (`testing/profile_training.py`) now runs the production-length ramp
  instead of one squeezed into its capped steps.
- Unreachable guards are deleted (EMA callback, curriculum, preprocessing, path helpers). Turning a
  path into a relative path for `preprocessing_summary.json`, the export metadata and the config
  snapshots no longer swallows errors: a path that cannot be resolved (e.g. a symlink loop) now
  raises instead of being written as given.

## REVISION 2026-09-30: named constants, explicit sentinels, figure style, provenance (audit theme T5)

Training numerics are unchanged (30-step fingerprint of both stages bitwise identical to the T7
reference). What changed:
- **Caps are explicit.** `preprocessing.pool_size` and `preprocessing.output_trajectories_per_file`
  are `null` in both configs: null means no cap (every source group of every raw file), replacing
  the 1e6 values that meant "all" but acted as caps. An integer is a deliberate per-file subsample;
  pool subsampling is now logged at WARNING (before, it was silent), and the output-cap warning says
  it counts written windows (4 per kept group). Per-file `n_groups`, `n_pooled` and `truncated` are
  in preprocessing_summary.json (T7). Outputs are unchanged while no cap binds (audit check on the
  synthetic set). The recorded preprocessing block of a processed dir built with 1e6 differs from
  the config, so src/main.py refuses such a build: rebuild it (the 4-file real build of 2026-09-30
  included).
- **Removed config keys** (now unknown keys): `runtime.log_every_n_steps` (every metric is logged
  per epoch; Lightning's step-logging interval is set to the epoch length, which the old 1e6 value
  was clamped to anyway), `runtime.checkpointing.monitor` and `training.scheduler.monitor` (could
  only be "val_loss"), and `training.scheduler.mode` ("max" was accepted and would have cut the LR
  while val_loss fell, against "min" checkpointing and early stopping). The monitored metric and
  direction are the constants `_MONITOR_METRIC = "val_loss"` / `_MONITOR_MODE = "min"` in
  src/trainer.py, shared by the plateau, checkpointing and early stopping. A pre-change
  `config.resolved.json` is refused by src/main.py until these keys are deleted.
- **Export batch cap.** `B_MAX = 16384` stays (per-key row); the export gate now also checks B = B_MAX
  and that B_MAX + 1 is refused, and the AOTI validation adds a B_MAX case (not run on daw).
- **AOTI validation inputs** are floored at the export's epsilon (the training floor, 1e-30), not at
  1e-15, so trace species down to the training floor reach the comparison (implemented, not run on
  daw).
- **Diagnostics.** The rollout figure plots a fixed window by default (first sample of the first
  validation shard; seeded random selection is opt-in), against "time since rollout start", with
  the shard, sample and start index in the legend. The loss figure reports the best val_loss and
  its epoch (best.ckpt's) and the last values, and draws the stage-2 training rollout horizon K on a
  right axis (train_loss is averaged over the curriculum K, val_loss always over
  `training.rollout_steps`). All figure scripts follow the user's figure rules with inline settings;
  `testing/science.mplstyle` is deleted.
- **Provenance.** The "Inherited / unjustified config values" table below lists every config value
  with no per-key evidence row, its v3 reference value and library default.

## REVISION 2026-09-30: per-epoch timing, throughput and job-log line (audit theme T4)

Training numerics are unchanged (30-step fingerprint of both stages bitwise identical to the T7
reference); only what is logged changes, and no sync is added per step.
- `epoch_time_sec` now uses `time.perf_counter` between device-synchronized hook boundaries and is
  documented as training plus the end-of-epoch validation pass (checkpoint writes come after it).
- `metrics.csv` gains `perf_val_time_s`, `perf_train_step_ms`, `perf_train_samples_per_s`,
  `perf_train_tflops` and (CUDA) `perf_peak_mem_gb`; definitions in the `src/trainer.py` docstring.
  Model FLOPs are 2*sum(in*out) over the `nn.Linear` layers per sample-step forward, x3 for a
  step with backward, x1 for a no-grad skip step. Epoch 0 includes compile and CUDA-graph warm-up.
- The job log gets one INFO line per epoch (epoch, K_train, LR at the epoch start and after its
  plateau step, train and val losses in `%.6e`, epoch time). EarlyStopping's 3-decimal messages and
  the checkpoint file names are unchanged.
- New diagnostics: `testing/efficiency_summary.py` (medians of the perf columns over every metrics
  segment, plus the launcher's GPU logger `logs/<run dir name>_gpu_util.csv`: utilization, SM clock,
  power-cap share, and achieved TFLOP/s against the TF32/BF16/FP32 peak at the measured clock) and
  `testing/profile_training.py` (torch.profiler summary and Chrome trace of a short fit, written to
  `<work_dir>_profile/`, never to the run directory).

## REVISION 2026-09-30: run contracts, fingerprints and provenance (audit theme T7)

Training side. Training numerics are unchanged: the 30-step fingerprint of both stages (live and EMA
weights, every training loss) is bitwise identical to the T1 reference; val_loss changes (first
item). What changed:
- **Validation is deterministic.** The validation set is a function of the processed data and one
  count only: the N_val x n_starts (trajectory, start) windows of the validation split are
  enumerated (n_starts = T - K = 99 at K=1, 90 at K=10) and an even subset of
  `dataset.val_samples` of them is scored (all when the split holds fewer), with no generator. It no
  longer depends on system.seed, the batch size, the epoch, the sanity batch, a resume or the
  preload device. With today's counts stage 1 scores an even stride of about 32 of the 99 starts
  per trajectory and stage 2 (92 >= 90 starts) scores every window (about 2% fewer validation
  batches than before, from the counts). val_loss is comparable across seeds and batch sizes on the
  same processed data, at equal rollout_steps and skip_steps (the skip mask still differs between
  recipes); val_loss histories from before this change are not comparable with later ones. Never
  resume (checkpoint_mode=resume) a work_dir started before this change: its restored best/ES/plateau
  state refers to the old windows. A resumed run's validation now equals an uninterrupted one's
  (audit check, both stages: every epoch's val_loss, scheduler and callback state). This supersedes
  the T1 notes below on validation windows and the sanity batch's draws.
- **Epoch length by count.** `dataset.train_batches_per_epoch` and `dataset.val_samples` replace
  `dataset.windows_per_trajectory` (an unknown key now). Their values reproduce today's epoch on the
  4-file build (per-key row); the epoch no longer grows with the dataset, so warmup, patience, ramp
  and EMA keep their length in optimizer steps when raw data is added unless these counts are edited.
  Startup logs the batches per epoch and every epoch-counted knob in optimizer steps.
- **Finished runs are marked.** A run writes `work_dir/COMPLETED` (atomic JSON: stop reason,
  epochs, global step, last val_loss, best checkpoint source, val_loss and sha256) only when it
  finished: max_epochs reached, or early stopping with patience exhausted, with a finite last
  val_loss and best score. That decision is made before best.ckpt is copied, so an interrupted or
  diverged run publishes neither and exits 1. A resume is refused while COMPLETED exists, and also
  when its checkpoint is from the last epoch of max_epochs or early stopping already stopped it (a
  run that ended before writing the marker). To extend a finished run deliberately, delete
  COMPLETED and raise `training.max_epochs`.
- **Stage-2 handoff (decision 2026-09-29): stage-1 `best.ckpt`.** `configs/stage2.json` loads
  `../models/stage1/checkpoints/best.ckpt` (best val_loss, EMA weights preferred), which replaces the
  last.ckpt handoff of the 2026-07-16 status note. weights_only also refuses a source run without
  COMPLETED (a best.ckpt of a run being extended is not a handoff). Clear any existing stage-2
  work_dir before the first run with this handoff.
- **Normalization fingerprint.** `norm_sha256` = sha256 of the canonical JSON (sorted keys, compact)
  of normalization.json's `normalization_methods`, `per_key_stats`, `epsilon`, `min_std`, `dt`,
  `species_variables` and `global_variables` (src/utils.py). Every checkpoint carries it in its
  hyperparameters. weights_only and resume refuse a checkpoint whose fingerprint is missing or differs
  from the processed_dir's, and weights_only rebuilds the loss statistics (criterion buffers) from the
  manifest instead of taking them from the checkpoint. Checkpoints made before this change have no
  fingerprint and cannot be handed off or resumed. Alternative 3 below was rewritten accordingly
  (and deleted on 2026-10-01, T8 entry).
- **Provenance.** Each invocation appends one line to `work_dir/provenance.jsonl` before training:
  time, config path and snapshot sha256, checkpoint_mode, the init checkpoint's path, sha256, epoch,
  step, optimizer LR and val_loss scores, git HEAD / dirty (untracked files ignored) / diff sha256,
  python/torch/lightning/numpy, CUDA and cuDNN, GPU name/UUID/driver, host, CUDA_VISIBLE_DEVICES,
  CPU affinity, thread settings, effective TF32, processed_dir and norm_sha256. Every field is
  best-effort (an error string, never a failed run).
- **Handoff baseline.** A weights_only run scores the loaded weights once on the validation set
  before training, on a separate logger-free Trainer, and writes `work_dir/init_val.json` (not in
  metrics.csv, not seen by early stopping or checkpointing).
- **TF32 is a config key** (`precision.allow_tf32`, per-key row); production behaviour unchanged.
- **Environment.** `requirements.lock` (the daw venv freeze; torch 2.8.0+cu126 by wheel URL) and
  `.python-version` (3.12.14) are the reproducibility source: `uv venv --python 3.12.14 && uv pip
  sync requirements.lock`. pyproject pins torch 2.8.* and lightning 2.6.* and is not installable as
  a package (no modules are built).

Data side (one preprocessing change, then one real build of the 4 raw files on daw, 2026-09-30:
584 s with 4 workers, peak 13.5 GB in the processed dir):
- **Window draws per trajectory.** Each trajectory's windows (dt and start time) come from its own
  generator seeded with [preprocessing.seed, a SHA-1 word of the group name]; the per-file
  generator (seeded with the file name, no longer the file's index in the sorted list) only shuffles
  the group pool. A window therefore no longer changes when a file is added or renamed elsewhere, or
  when another group is kept or rejected. This redrew every window once (a training-data change:
  any earlier processed dir and every checkpoint made on it are superseded); split membership,
  counts and rejects are unchanged. Verified: with only the two seedings reverted, the new code
  writes shards equal to the previous code's array by array.
- **Split seed.** Split membership uses the module constant `SPLIT_SEED = 1234` (the historical
  preprocessing.seed, so today's splits are unchanged), not preprocessing.seed, so a sampling
  replicate keeps the same test trajectories. The split key (file basename and group name; renaming
  a raw file moves about a third of its groups to another split) and the raw basenames are recorded
  in the summary; a group name present in two raw files fails the build. Keying on the group name
  alone (rename-proof) is a pending author decision.
- **Build record and checks.** preprocessing_summary.json now holds one row per raw file (name,
  size, groups, rejects, windows per split, shards, raw time grid), group-level rejects
  (non_finite, non_positive_global, too_few_valid, drop_below; whole source groups) and chunk-level
  rejects (no_fit_chunk, interp_non_finite; window attempts) with groups_seen = sum(group rejects) +
  groups_kept, `windows_written_by_split` (was `counts_total`), the (T, log10 P) coverage table,
  start-time and dt histograms, the config blocks used, norm_sha256, and code/library versions.
  src/main.py now refuses to train on a processed dir whose summary is missing or predates this
  record, whose recorded preprocessing / normalization / data blocks differ from the config's
  (except overwrite and num_workers; per-key "config vs built" message), whose normalization.json
  is not the one the summary describes (fingerprint), or (fresh runs) whose raw files were added or
  resized since the build; removed raw files or a missing raw_dir only warn (raw data may be pruned
  for quota). It also validates the manifest's statistics with named errors.
- **Manifest additions (descriptive; not part of the fingerprint).** `units` (P dyn/cm^2 (barye),
  T K, dt s, species: mixing ratios n_i / n_total over all gas of the generator run (inferred; the
  generator config is not on daw), the training initial gas held only the tracked species, outputs
  use the same denominator); `window_start_time_s` (earliest train window start: 0.001 s);
  `coverage` (train windows per 100 K x 1 dex cell, lowest-retention cell); `species_sum` (the
  train-split distribution of the tracked-species sum per state: min 0.0253, 0.1% / 1% / 5% levels
  0.6467 / 0.9109 / 0.9877 at 1e-4 resolution, median 1.0); `std_floored` (fields whose variance
  fell below min_std**2; none on the 4 files; an exactly constant column is not caught by it).
- **Measured on the 4-file build (recompute as data arrives):**
  - drop_below (rule kept) rejects 244,756 of 893,840 finite groups (27.4%), skewed by T and P:
    67.1% at 300-600 K, 54.0% at 600-900 K, 14.1-16.3% at 1200-2700 K, 25.4% at 2700-3000 K;
    37.6% at log10 P (barye) 0-1 down to 17.9% at 9-10. Kept groups per K are 2.6x denser at
    1200-2400 K than at 300-600 K. Worst 100 K x 1 dex cell: 700-800 K at 1e9-1e10 barye keeps 144
    of 3,297 (4.4%); every one of the 270 cells keeps >= 144 groups. Triggers (a group can have
    several): O 50.7%, C2H2 46.0%, OH 18.7%; 64.4% of rejected groups hold an exact zero. The
    per-cell table is in preprocessing_summary.json; the composition skew (kept cool groups are
    more O-rich, rejected hot groups are the extremely O-rich ones; audit R1-018) is not visible in
    it. Reweighting, a per-cell evaluation gate or a composition-aware out-of-distribution flag is
    the author's decision.
  - Per file: drop_below rejects 29.2-29.4% of run10017, run10142 and run9095 but 21.5% of run6050,
    whose initial compositions come from a different distribution (audit: median trace log10 y0
    about -10.1 vs -12.6; inferred to be a different sampler setting; the raw files carry no
    generator settings). All four files store the same grid: t=0, then 1e-3..1e8 s, 100 points.
  - Window starts: draw 0 of every kept group is anchored at the first positive raw time (1e-3 s),
    draws 1-3 are log-uniform, so 25% of windows start at exactly 1e-3 s and 31.8% of starts fall in
    [1e-3, 1e-2) s against 6.8% per later decade. Whether to keep this weighting is the author's
    decision.
  - Training states start at max(t_min, first positive raw time) = 1e-3 s: the raw t=0 rows (the
    generator's randomized initial compositions) never enter training (log-time interpolation cannot
    represent t=0). Unrelaxed compositions (under ~1 ms after initialization) are therefore out of
    distribution, and the model cannot detect that from its inputs. A future fixed-dt evaluation
    generator must start at k >= 1 (prepare_log_interp maps t=0 to y(1e-3 s)).

Export and evaluation side (contract changes for hosts; the network is unchanged):
- **Out-of-range rows are NaN (approved policy).** The exported step returns NaN in every species
  for a row whose dt, P or T lies outside the train-split box by more than `OOR_TOL = 1e-6` of the
  range (log10 for dt and log-* globals), or whose inputs are not finite. In-range rows are bitwise
  identical to the previous export (audit check on CPU and CUDA: seeded in-box inputs, validation
  states, the exact box edges). On the 4-file data 0 of 64,426 validation and 0 of 65,046 test
  trajectories lie outside the train box (audit pre-count, re-checked on the T7 build). Species are not range-checked (values
  <= epsilon are floored, as before; the author's decision), and outputs are neither clipped nor
  renormalized. Hosts (the VULCAN Emulator wrapper, Canoe) must handle NaN rows; an installed
  model.pt2 made before this change keeps clamping until it is replaced.
- **Export metadata, version 2** (schema change): adds `units`, the guard's exact float32 box
  (`oor_box`, `oor_tol`, `oor_policy`), `per_key_stats`, `window_start_time_s`, compact (T, P)
  coverage and the lowest-retention cell, `species_sum`, `norm_sha256`, the checkpoint's sha256,
  weights source, epoch, step and score, the training TF32 setting, the Lightning version and the
  export code's git commit; `cuda_visible_devices` is removed (slot-dependent; the external VULCAN
  loader reading it is unverified, author to confirm). `format` is unchanged.
- **Export contract.** The run is `paths.work_dir` of `AUTOCHEM_CONFIG_PATH`'s config (no edited
  RUN_DIR; one resolver, testing/run_dir.py, for every testing tool); it must hold COMPLETED and its
  checkpoint must carry the processed data's norm_sha256; every requested device must be available
  (no CPU fallback). Each artifact is written under .partial/, reloaded and gated on its device
  (in-range rows finite and within `EXPORT_PARITY_RTOL = 1e-6` of the eager wrapper, bitwise on CPU;
  out-of-range rows NaN) before it replaces the previous file. Example inputs are seeded and node
  stack traces stripped, so re-exporting a checkpoint reproduces the file's sha256 with the same code
  commit and torch build.
- **AOTI package (implemented, not run on daw).** Validation compares the package with the fresh
  export at dt_min, 6.4 s and dt_max plus one out-of-range row: identical NaN rows (equal to the
  inputs' out-of-range rows), no inf, and max |dlog10| <= `MATCH_MAX_DLOG10 = 1e-4` (about 10x the
  float32 AOTI-vs-eager noise measured on a 10x2048 random-weight proxy, 9.5e-6 dex; re-check on the
  first trained model). The package is installed only on a pass; otherwise both reports are written
  and the script exits 1. It is loaded once (default thread-safe mode), its example batch is floored
  at 2 (torch.export specializes size 1), it records its build target (machine, -march=native
  resolution, AVX-512/AMX flags, compiler, torch) and build threads, and its metadata is visible to
  the loader. The package does no input validation; the host contract (CPU float32 C-contiguous, one
  B in the dynamic range) is in its metadata and spec 4.3, with a Python guard to copy. It must be
  built on the machine that runs it (or the same CPU generation, OS/glibc and torch).
- **Rollout evaluation.** eval_rollout_error.py scores a seeded sample of the validation split
  (4000 sources, one window each; the test split once, for the final paper number), reports
  per-species dex p50/p90 and signed bias per step (recommended headline; author confirms) with
  bootstrap CIs, keeps the legacy pooled fractional percentiles (floor `FRAC_ERR_FLOOR = 1e-12`,
  inherited; 24.3% of scored truth lies below it on the 4-file data, audit measurement) without the hard-coded sibling
  numbers, ranks diverged entries as +inf error (fixed population), drops out-of-box sources before
  the rollout (counted apart), and checks that its physical inputs re-normalize to the shard values.
  Its rows are random-dt windows with K <= 98; a fixed-dt K >= 100 evaluation set does not exist
  yet. predictions.py also defaults to the validation split.
- **Latency benchmark.** bench_speedup.py draws its inputs inside the trained box, times every batch
  size per (device, threads) in interleaved rounds, reports the per-cell cost at the host batch
  (`DEPLOY_BATCH_SIZES = (150,)`, inherited from the VULCAN HD189 example nz, unconfirmed for the
  VULCAN-0D reference and Canoe) and the peak throughput separately, prints mini-chem ratios only at
  mini-chem's 1 thread (cross-host: where mini-chem was timed is unrecorded), and exits non-zero
  instead of writing a partial CSV.

## REVISION 2026-09-30: correctness and contract fixes (audit theme T1)

Production training numerics are unchanged by these fixes except the three listed at the end of this
entry: without those, the 30-step fingerprint of both stages (live and EMA weights, every training
loss, val_loss) is bitwise identical to the pre-audit baseline,
preprocessing outputs (synthetic set and a real-data subset) compare equal array by array, and exports
of the smoke checkpoint give identical weights and outputs (the graph loses five no-op denominator
clamps; the metadata gains the checkpoint's `model` block). What changed:
- Configs are checked harder before anything is written or loaded: duplicate JSON keys and
  NaN/Infinity, `_x` keys other than `_comment*`, dotted raw keys (`"a.b": v`; write the nested
  block), wrong value types (e.g. `400.0` epochs, `"false"`), value ranges (max_epochs >= 1,
  optimizer/scheduler ranges, gradient_clip_val null or > 0, save_top_k -1 or >= 1), cross-key
  curriculum/skip/checkpointing rules, and duplicate species names (config, manifest, export).
  `runtime.log_grad_norm` and `preprocessing.num_workers` are now required keys.
- Early stopping and the LR plateau observe the same epochs: both start counting at the first epoch
  after the LR warmup and the curriculum ramp (training behaviour; per-key row below).
- Run directory: the training process flocks `work_dir` (a second launch on a live run fails at
  startup) and writes `train.log` there (appended on resume); a `work_dir` holding only `train.log`
  counts as empty for a fresh start. A resume must continue the same run: its checkpoint in this
  run's `checkpoints/`, ModelCheckpoint state written there (a moved or copied run dir is refused;
  restart it with `checkpoint_mode="weights_only"`), and a config equal to `config.resolved.json`
  except `training.checkpoint_mode`, `runtime.checkpoint` and `training.max_epochs`; a resume writes
  `config.resume[.N].json` beside the untouched `config.resolved.json`. `last.ckpt` always holds the
  latest completed epoch. A non-finite val_loss aborts the run (exit 1) before that epoch's checkpoint
  is written and publishes no best.ckpt (finite top-k and last.ckpt files remain); launchers must gate
  on the exit status, not on "any *.ckpt". SIGTERM exits 143 and best.ckpt is not published.
  weights_only refuses non-finite weights, a checkpoint whose model block differs from the config's
  (comment keys ignored), and a missing `ema_state_dict` when the source run used EMA.
- Preprocessing: one run per processed dir (`<processed_dir>.lock`, a persistent 0-byte sibling
  file); the previous dataset stays intact until the new one is complete (built in `_tmp_final/`,
  swapped in with `normalization.json` last); each physical shard is deleted once its final shard is
  written; the published row counts must equal the sampled counts; worker processes die with the
  parent (Linux PR_SET_PDEATHSIG). P/T: a non-scalar value fails the run (schema mismatch); a
  non-finite one or one that is not > 0 as float32 rejects only that trajectory group, counted as
  `non_finite` / `non_positive_global` in `preprocessing_summary.json` (no current group is affected:
  all have P >= 1.0001, T >= 300.002, audit measurement). `drop_below >= epsilon` is required.
- Export: only Lightning checkpoints are accepted (keys under `model.`/`criterion.`), EMA weights are
  required when the run used EMA, the checkpoint's model block must equal the config's, weights must
  be finite and every normalization statistic is validated (NaN-safe; stds >= min_std, ranges hi > lo
  as float32); artifacts are written under `.partial/` and renamed into place.

Three further T1 fixes change training or validation numbers slightly:
- The training batch stream is reseeded at every epoch from (system.seed, epoch), so a resumed run
  draws the same batches as an uninterrupted one (audit check, both stages: weights, EMA weights and
  optimizer state after 2 + 2 resumed epochs equal those of one 4-epoch run bit for bit). Fresh-run
  batch sequences change once. Validation windows still depend on the validation stream's history,
  so a resumed run validates on different windows (and its ES/plateau/checkpoint decisions can
  differ) until validation is made deterministic (audit theme T7).
- One validation batch runs before epoch 0 (Lightning's sanity check), so a compile or shape failure
  in the validation path surfaces in minutes instead of after the first epoch. It is not an epoch (no
  LR step, no checkpoint, no ES count), but it consumes validation-stream draws, so val_loss values
  are not comparable with runs made before this change.
- The EMA averages parameters only and copies buffers (loss statistics, Fourier frequencies).
  Averaging a constant drifted it (2.9e-5 absolute on the loss log-means after 30 steps), so the
  EMA-evaluated val_loss moves in the 6th to 7th digit; the parameter EMA is unchanged bit for bit.

## REVISION 2026-09-30: per-step training overhead cut; stage-1 CUDA graphs (audit theme T3)

Training numerics are unchanged: the 30-step fingerprint of both stages (live and EMA weights, every
training loss, val_loss) is bitwise identical to the pre-audit baseline. What changed:
- Three run constants are no longer logged on every training step; each call forced a GPU sync (3 per
  step). `metrics.csv` loses the `train_skip_steps` and `train_detach_between_steps` columns (the settings
  are in `hparams.yaml`), and one-jump (stage-1) runs also lose the constant `train_rollout_steps`.
  Autoregressive runs log `train_rollout_steps` once per epoch; it follows the curriculum.
- The epoch training losses are summed on the GPU and logged once per epoch. The logged values are
  identical to the last bit at the shipped batch size.
- The EMA update runs as grouped `torch._foreach` calls over cached tensor lists (same arithmetic).
- Stage 1 compiles with `runtime.torch_compile.mode = "reduce-overhead"` (CUDA graphs); see the per-key
  table.

## REVISION 2026-09-30: dead options and code paths removed (audit theme T6)

Nothing the production configs do changed: a 30-step training fingerprint of both stages
(live and EMA weights, every training loss, val_loss; synthetic production-shaped data) is
bitwise identical before and after, and preprocessing outputs (synthetic data and a small
real-data subset) and exported artifacts (the smoke checkpoint) compare equal array by array.

Removed config keys (a config that still sets one now fails as an unknown key):
`precision.dataset_dtype`, `normalization.globals_default_method`, `runtime.load_weights_strict`
(weights_only loading is always strict), `runtime.mode`, `dataset.preload_to_device` (every split
is preloaded), `dataset.shard_cache_size`, the whole `model.autoencoder` block,
`training.num_workers`, `training.persistent_workers`, `training.prefetch_factor`,
`training.scheduler.type`, `training.scheduler.min_lr_ratio`, `training.curriculum.mode` (linear
only), and the `preprocessing.dt` / `preprocessing.dt_mode` tombstones. The ReduceLROnPlateau keys
(`factor`, `patience`, `threshold`, `min_lr`, `mode`, `monitor`) are now plain required keys. A
`config.resolved.json` written before this change must have these keys deleted before
`src/main.py` accepts it as a config; `testing/export.py` reads only the data/model/paths blocks
and still exports older run directories.

Removed code (recover from git history, tag `pre-audit-2026-09-29`):
- The autoencoder model family and the unused multi-step `FlowMapMLP.forward` (see alternative 4).
- The map-style (non-preloaded) dataset path. Its 2-shard LRU cache thrashes under random access
  (audit measurement: about 17 s and 43.6 GB read per 2048-sample batch at 700 shards, against
  0.46 ms for the preloaded stream), so it was never a scale fallback. There is now no loading
  path for data larger than RAM; that is an open design item. The shards are uncompressed
  `np.savez` ZIP archives, so their members could be opened with `np.memmap` at their data
  offset (an input for any beyond-RAM loader); compressing them (`np.savez_compressed`) would
  give that up and is undecided.
- The `cosine_with_warmup` scheduler. The v3 reference run used it
  (`git show 09748d7:models_old/v3/config.resolved.json`: `cosine_with_warmup`,
  `min_lr_ratio` 0.01, `warmup_epochs` 30, `max_epochs` 300), so that schedule can no longer be
  re-run from this repo. So did the v2_bigger_batch run
  (`git show 09748d7:models_old/v2_bigger_batch/config.resolved.json`: `warmup_epochs` 50,
  `max_epochs` 1000), and both reference runs also trained with `gradient_clip_val` 0.0 and no EMA
  or early stopping, so neither can be reproduced from this repo (added 2026-10-01).
- Mac-era code: MPS device and export targets, and the `KMP_DUPLICATE_LIB_OK` shim (the venv's
  only OpenMP runtime is GNU libgomp, which does not read it).
- The bf16 export option (exports are float32 only; see "Verification performed" below).
- Duplicate diagnostics: `testing/benchmark.py` and `testing/bench_eager_devices.py`
  (`testing/bench_speedup.py` is the one latency benchmark and times the CPU and CUDA exports),
  and `processing/testing_data.py` (preprocessing's `read_time` already hard-fails on its checks).

## REVISION 2026-07-16: `output_trajectories_per_file` restored to **1e6**

Commit df2c44f (2026-06-17, "fix to multi-thread processing") accidentally cut
stage-1 `preprocessing.output_trajectories_per_file` from 1e6 to 1e5 while adding
`num_workers: 64`, contradicting the "consume all" user decision in the table below,
the config's own inline comment, and stage2.json. Restored to 1e6; both configs now
carry identical preprocessing blocks except `overwrite` (stage-1 true, stage-2 false),
with `num_workers: 64` added to stage2.json for parity (superseded 2026-09-30: 16 in both
configs, see the per-key table; and both caps are now `null` = no cap, T5 entry). If a processed dataset was
already built on the HPC after df2c44f, check the recorded value in the processed
dir's `preprocessing_summary.json` and rebuild if it was made with the 1e5 cap
(the launcher, now run_daw.sh, skips preprocessing whenever the processed dir's normalization.json exists).

Stage-1 -> stage-2 checkpoint handoff: DECIDED 2026-09-29, stage 2 initializes from stage-1
`checkpoints/best.ckpt` (best val_loss, EMA weights preferred), published only by a finished stage 1
(see the T7 entry above). `testing/export.py` also exports `checkpoints/best.ckpt`, matching the
val_loss selection contract (spec 1.3 / 6.2). Correction 2026-09-30 (audit T1): with Lightning 2.6
the top-k callback's `save_last` refreshes `last.ckpt` only in epochs that enter the top-k
(`lightning/pytorch/callbacks/model_checkpoint.py` 494-498), so `last.ckpt` could lag the final
epoch; a separate monitor-free callback now writes it every epoch, so "last.ckpt" means stage 1's
true final epoch.

## REVISION 2026-06-11: `fourier_dt.enabled` flipped to **false**

The F1/F2 production-mirror trunk shootout (~130k params — 17× the original screening
scale — exact Auto-Chem trunk semantics, 2 seeds, plus an LR probe) found that
**Fourier-Δt features on top of the residual+LN trunk are harmful**: worst
fixed-dt-mean error in both seeds (0.84/1.30 dex vs 0.45/0.42 for the same trunk
without Fourier), erratic across seeds on the geometric metric (2.7e-2 / 4.8e-3), and
not rescued by lowering LR. The clean 2×2: no stabilizer = collapse; residual skips
OR Fourier alone = good; both = interference. The original 6.2× Fourier win (E1) was
measured on plain trunks at 7.6k params; at ~130k params Fourier still helped the plain trunk
(the fallback below) and interfered only with the residual skips, so this is a trunk-dependent
Fourier-dt effect measured at one scale, and its scale dependence is untested (corrected
2026-10-01; the earlier text called it a small-scale result that did not transfer; that E1's
plain trunk and F1/F2's plain-LN trunk are the same non-residual family is inferred). It is the
failure mode the §"Evidence base" policy anticipated: a sandbox win that did not carry over to
the production trunk. Because the feature was flag-gated, the correction is this
config flip; no code changes. The `FourierDtFeatures` code remains in src (validated,
export-safe) for the fallback below and the production-scale spot-check (candidates pending the
author, see "Production-scale spot-checks").

Consequences:
- Primary architecture = **the v3-lineage trunk unchanged** (10×2048 residual+LN+SiLU,
  predict_delta, plain scalar dt) — also the most consistent arm on the deployment
  metric and the deployed trunk; one of two trunks with a recorded run at this data scale (the
  other is v2_bigger_batch's 6×2048, see "Where the evidence is"; corrected 2026-10-01, was
  "the only trunk validated at true production scale historically").
- New fallback (won the geometric metric both seeds, 1.3e-3/2.3e-3): plain-LN trunk
  WITH Fourier — under `model.mlp` set `"residual": false` and under `model.fourier_dt` set
  `"enabled": true` (nested keys; a dotted raw key is rejected at startup since 2026-09-30).
  Caution: that trunk collapsed without Fourier (E7), so the pair must be flipped together.
- Depth confirmed: shallow-wide (4×198) lost on both metrics → keep 9–10 layers.
- Caveat carried forward: the G2 single-vs-banded result was measured with
  Fourier-equipped models; its mechanism (dt-diverse data as augmentation) is
  architecture-independent, but the banded comparison was not re-run with the
  no-Fourier trunk.
- The fourier_dt sigma/num_freqs rows below now apply only to the fallback variant.

Date: 2026-06-09 (revised 2026-06-11; configs flattened/renamed and band/fallback configs removed 2026-06-15).
Scope: the autoregressive production configs (`configs/stage1.json` + `configs/stage2.json`),
the two src changes that support them (`model.fourier_dt`, `loss_discount_gamma`),
and the validation protocol. Campaign log with full numbers:
`Chemulator_Project/EXPERIMENTS.md` (the former `notes.md` was consolidated into it
on 2026-06-15). Experiment artifacts: the E-series run directories
(`Robertson_Tests/autoregressive/runs_e*`) were deleted in the 2026-06-15 cleanup
after their numbers were verified and preserved in EXPERIMENTS.md; the surviving
run dirs are `runs_f*` and `runs_l*`.

**Where the evidence is (checked on daw 2026-09-29, audit R1-010; added 2026-10-01).** None of
EXPERIMENTS.md, IMPLEMENTATION_PLAN.md, `Robertson_Tests/` (`runs_f*`, `runs_l*`, `figures_l1`),
the mini-chem timing driver (`bench_dlsode.f90`, `bench*.nml`), `models/v3` or `models/0` is in
this repo or in the user's space on daw, and none was ever committed here. Every sandbox number in
this file (E1–E7, F1/F2, G2) is therefore quoted from this file's own summaries and is
unverified until those records are recovered from the old machine (an author action; open items).
The only prior-run records in git are six files of two runs,
`git show 09748d7:models_old/{v3,v2_bigger_batch}/{config.resolved.json,hparams.yaml,metrics.csv}`
(added 09748d7, 2026-03-27; deleted f2c1613, 2026-03-30). That models_old/v3 is the deployed v3 is
inferred (directory name; its work_dir is `../v3_even_smaller_time_range`). From those files:
- v3: 10×2048, batch 8192, dt [1, 50] s, n_steps 300, 100 windows per trajectory,
  samples_per_source 1, 5000 trajectories per file, `cosine_with_warmup` (warmup 30, 300 epochs),
  clip 0, no EMA, no early stopping. val_loss first reached 6.98e-4 at epoch 45, its minimum is
  4.03e-4 at epoch 207, and the last logged value is 4.20e-4 at epoch 214 (of max_epochs 300).
- v2_bigger_batch: 6×2048, batch 8192, dt [0.05, 5000] s, lr 1e-4, wd 1e-4, the same schedule
  family (warmup 50, 1000 epochs), clip 0, no EMA or early stopping. val_loss was 8.8e-3 to
  1.08e-2 only in epochs 0-7 of its 50-epoch warmup, first reached <= 7e-4 at epoch 60, and has its minimum
  4.11e-4 at epoch 488 (last logged epoch 501).
The val_loss formula is unchanged since then (audit), but the data and validation windows differ,
so neither run's val_loss is comparable with today's (spec 6.7).

## Evidence base and its limits

Decisions come from three sources: (a) the **local Robertson campaign**
(2026-06-09; 3-species stiff ODE sandbox, ~7.6k-param models, 2k–10k trajectories,
seeds 0–2, CPU), (b) **adversarially verified literature** (every load-bearing claim
checked against primary sources; the List et al. numbers were wrong until 2026-10-01, see Key
citations), and (c) **prior Auto-Chem/Chemulator runs**
(v2/v3 lineage, the published paper).

Two evaluation metrics were used throughout; they disagree in important ways:
- **geometric rollout** (consecutive log-grid steps, the sandbox's native task; dt grows along
  each rollout because the grid is log-spaced, inferred from the wording): the *discriminator*.
  Seed spread varies by arm (corrected 2026-10-01; the earlier text gave one ±3% for the whole
  metric): ±3% for E3 K=10 detached γ=0.9 (3 seeds), 15% for E4 BPTT γ=0.9 (2 seeds), 1.8–5.6×
  for the F1/F2 pairs. Compare each arm against its own spread.
- **fixed-dt rollout** (constant dt for 100 steps — what Canoe/VULCAN actually do):
  seed noise ±25% — the *catastrophe detector*. One-step/val metrics were NOT used
  for selection (Spearman ρ(one-shot, rollout) = 0.41).

**Transfer caveat (applies to every "local" row below):** production is ~37.9M params
(37,896,204 for the 10×2048 residual trunk, logged as `params=` by src/main.py; 37,931,020 for the
Fourier/non-residual fallback; corrected 2026-10-01, was "~100M"), 12 species conditioned on
(P, T), trained on the processed windows counted in the processed dir's
`preprocessing_summary.json` (`windows_written_by_split`; the raw-file count is the preprocessing
log's `files=N` line). The earlier "~62M samples" is dropped: it fits no reading of the local
yield (the 4-file build has 2,078,448 train and 257,704 validation windows) and cannot be checked
on daw. The sandbox is none of those. Per the agreed policy, local wins were adopted only when replicated across seeds, mechanistically
sensible, AND cheap to disable; close calls ship both options. Sandbox models also bake
a per-step simplex projection into the rollout (production does not) — sandbox
*stability* numbers partially lean on it; relative comparisons should transfer better
than absolute ones. The dt schedule differs too (added 2026-10-01, audit R4-012): the geometric
metric's rollouts used growing dt (consecutive log-grid steps), while every Auto-Chem training,
validation and evaluation rollout keeps one dt for the whole window (preprocessing writes one dt
per window), so a rollout whose dt changes between steps is never trained or validated here.
Scale checks: 18× params (7.6k→135k) and 5× data (2k→10k) showed no inversion of any adopted
finding (as of 2026-06-09; two days later F1/F2 reversed the Fourier-dt adoption on the residual
trunk, see the 2026-06-11 revision).

## Confidence labels

- **high** — replicated locally + literature-aligned + scale-checked.
- **medium** — replicated locally (2–3 seeds) at sandbox scale; untested at production scale.
- **low / inherited** — carried from the v3 run config, the dt-band templates, or literature
  alone; not discriminated locally. Each such row needs a spot-check or fallback (see
  "Production-scale spot-checks") or an explicit waiver, pending the author.

Labels the table also uses (rubric extended 2026-10-01, audit R4-004; the author has not
re-scored the rows):
- **medium-high** — not defined by the rubric: read it as "medium" (sandbox-replicated, untested
  at production scale) plus literature agreement. Keep it with this meaning or drop it (author).
- **decision** — a user or author choice, not an empirical result.
- **correctness fix** — a value forced by the code (anything else fails), not evidence.
- **production-scale evidence (prior runs)** — prior Auto-Chem/Chemulator runs; the only ones in
  the repo are the two git-history runs under "Where the evidence is".
- **n/a** — throughput or contract settings with no accuracy effect.
Rows whose own text does not support their label carry a bracketed "[rubric: ...]" note.

## Src changes (both flag-gated; absent key = exact legacy behavior)

1. **`model.fourier_dt`** (`src/model.py: FourierDtFeatures`): fixed, seeded random-Fourier
   embedding of the normalized scalar dt, concatenated as `[dt, sin(2πB·dt), cos(2πB·dt)]`
   — a strict superset of the legacy input. Wired into FlowMapMLP. torch.export-verified.
2. **`training.autoregressive_training.loss_discount_gamma`** (`src/trainer.py`):
   per-step weight γ^(k−skip) on the rollout loss, both detached and BPTT paths.
   Default 1.0 reproduces legacy math exactly (verified: weights/normalization identical).

## Primary config: per-key evidence

| Key | Value | Evidence | Confidence |
|---|---|---|---|
| **layout: ONE dt-conditioned model** (not dt bands) | single config pair over [0.1, 1e3] s | **Gate G2 (E2)**: band specialists collapsed at deployment-style fixed-dt *inside their own bands* (low band @ dt=0.01 s: 1.3–12.3 dex vs single model 0.33; high band @ dt=100 s: 1.8–2.0 dex vs 0.28–0.29; both seeds) while winning only marginally on their native grid task (2.09e-3 vs 2.41e-3, no handoff spike). Mechanism: dt-diverse training is augmentation against the (state, dt) distribution shift AR deployment creates. Literature: specialists' only proven benefit (Pangu-Weather HTA, HiTS) is composing large steps — unreachable at host-fixed dt; stitched specialists show documented boundary discontinuities (Bonavita 2024); single conditioned models proven over ≥7 time decades in chemistry (Asensio Ramos 2024) and lead times (Stormer). Consistent with the v3 narrow-dt-window deployment bottleneck. | medium-high (caveat: sandbox bands conflate time-segment with dt-band; production pathology likely milder → no banded fallback: configs removed 2026-06-15 (f335fed); not an approved fallback (corrected 2026-10-01). G2 was measured on Fourier-equipped models and not re-run on the no-Fourier production trunk) |
| `model.fourier_dt.enabled` | **false (REVISED 2026-06-11)** | Original adoption (E1: 6.2× at 7.6k params, 3 seeds) was measured on PLAIN trunks. F1/F2 at ~130k params on the exact production-mirror residual trunk: Fourier+residual is the worst arm (fixed-dt mean 0.84/1.30 vs 0.45/0.42 without; geometric erratic 2.7e-2/4.8e-3; LR probe no rescue, 2 seeds). Either stabilizer alone works; together they interfere: a trunk-dependent Fourier-dt effect (it interferes with residual skips at ~130k params); its scale dependence is untested. See REVISION header. The spot-check list below holds the production-scale candidates (pending the author). | medium-high (replicated at the most production-like local scale) |
| `model.fourier_dt.num_freqs` | 16 | Tancik 2020: width secondary, saturates; 16–32 ample for a smooth scalar conditioner (diffusion models use 128 for far harder targets). Sandbox used 8 successfully; 16 adds margin at negligible cost. Applies only to the Fourier-dt fallback. | medium (literature) [rubric: low, literature only; waiver pending the author] |
| `model.fourier_dt.sigma` | 1.0 | EDM2 pattern: normalize the log-quantity to O(1) (Auto-Chem's dt_norm ∈ [0,1] already is), then σ≈1. Sandbox win used σ=3.0 on the same normalization — both in the working range; σ too high risks wiggly dt-interpolation at the deployment dt (Tancik). Applies only to the Fourier-dt fallback; its σ sweep is listed once, under the spot-checks (on the fallback trunk). | medium (literature; exact value not locally discriminated) [rubric: low; spot-check candidate] |
| `model.mlp.hidden_dims` 10×2048, residual, LN, SiLU, `predict_delta` | unchanged from v3 lineage | v3 (10×2048) is the deployed trunk. Its recorded val_loss (git history, "Where the evidence is") has a minimum of 4.03e-4 at epoch 207 and ends at 4.20e-4 (epoch 214 of 300), on other data and validation windows (dt [1, 50] vs [0.1, 1000] s, n_steps 300 vs 100, windows 100 vs 32, samples_per_source 1 vs 4, 5000 vs all trajectories per file, batch 8192 vs 2048, no EMA), so it is not comparable with today's val_loss (corrected 2026-10-01; was "reached val ~7e-4", which is v3's first crossing at epoch 45). LN value re-confirmed locally (MLP+LN 2.9× over MLP, E1); delta-head small-init is the sandbox/paper convention. Not re-tuned: out of local scope. The latent-linear sandbox result (3–5× better than the MLP per EXPERIMENTS.md, not on daw) is not adopted here; that architecture exists as LatentLinearFlowMap in the sibling repo imalsky/Chemulator (src/model.py, added in efa5378), not in Auto-Chem, and the architecture decision is deferred (author). | inherited |
| `preprocessing.dt_min/dt_max` | 0.1 / 1000.0 s | User deployment decision. Canoe dt≈6.4 s is 1.8 decades above dt_min and 2.2 decades below dt_max (corrected 2026-10-01; was "~1.3 decades inside each edge"); a learned model is unreliable outside its training range, so the deployment dt sits well inside it (the Tancik "Fourier features interpolate well inside support" argument applies only to the Fourier-dt fallback). | decision |
| `preprocessing.samples_per_source_trajectory` | 4 (was 1) | E2 mechanism: per-state dt diversity is the robustness lever; 4 (dt, t_start) draws per source trajectory maximize (state, dt) coverage from the same raws. dt is log-uniform in every draw; draw 0 starts at the first positive raw time (1e-3 s; 25% of windows), draws 1-3 at a log-uniform time (T7 entry). | medium |
| `preprocessing.output_trajectories_per_file`, `pool_size` | null = no cap (REVISED 2026-09-30, was 1e6) | "Use the full dataset" (user decision); the full-set counts are in the processed dir's preprocessing_summary.json (the earlier "~62M samples across ~160 raw files" is dropped, 2026-10-01: unverifiable on daw and inconsistent with the local yield). The 1e6 values meant "consume all" but were caps: the output cap counts written windows (4 per kept group), so it would bind at 250,000 kept groups per file: 1.1x today's 224,000 groups per file with no rejects, about 1.4-1.6x at the measured keep rates (audit). An integer is a deliberate, logged and recorded subsample. | decision |
| `training.batch_size` | 2048 | Inherited from the dt-band templates (2048 since 960e467, 2026-03-24). "Paper finding: >2048 degraded generalization" is unverified (the paper is not on daw). Both recorded prior runs (v3 and v2_bigger_batch, "Where the evidence is") trained at 8192 and neither isolates batch size; v2_bigger_batch was near 9e-3 only during its warmup and reached a minimum of 4.11e-4, so the earlier "v2_bigger_batch (8192) plateaued at val ~9e-3" is deleted (corrected 2026-10-01). No Auto-Chem batch-2048 run has metrics in the repo. | inherited (dt-band templates); no in-repo run evidence (was "high (production-scale evidence)") |
| stage-1 `lr` / `wd` / scheduler / EMA 0.9995 / early-stop | template values | lr 3e-4, wd 1e-5, warmup 30 epochs and the loss weights are v3 run-config values (v3 ran them under `cosine_with_warmup`, see REVISION 2026-09-30). Plateau patience 15, min_lr 1e-7, EMA 0.9995, early stopping 30 (then with min_delta 1e-5) and clip 1.0 come from the dt-band templates at 0bcdffc (2026-05-09), 32 windows per trajectory from d92d930 (2026-05-10), threshold 1e-3 and max_epochs 400 from 9738917 (2026-05-14). No in-repo run record uses this scheduler/EMA/early-stop/clip combination (corrected 2026-10-01; was "v3-lineage ... converged to val ~7e-4 historically"). weight_decay 1e-5 is exactly inactive in float32: AdamW decays a weight by lr·wd per step (3e-9 at the stage-1 peak, 3e-10 in stage 2), below float32's relative half-step 2^-25 ≈ 3e-8, so every decay step rounds back (audit: 300 real-model steps bitwise identical to wd = 0 on CPU; CUDA fused inferred from the kernel source); the 3.9% / 0.56% whole-run shrinkage bounds hold only in float64 (choice open, see open items). EMA window rule (1/(1−d) ≈ 1–10% of updates; whether per epoch or per run was never stated, and the two readings disagree, as the shares that follow show; ACE uses 0.9999): 0.9995 is a 2,000-step horizon, 0.06 of a 32,475-step stage-1 epoch and 0.015% of the 400-epoch budget (0.999 in stage 2: 1,000 steps, 0.05 of a 20,000-step epoch); 0.9999 would be 10,000 steps; not locally discriminable. | inherited [no spot-check; waiver pending the author] |
| `dataset.train_batches_per_epoch` / `dataset.val_samples` | stage 1 32,475 / 8,246,528 (since 2026-09-30, replacing `windows_per_trajectory` 32); stage 2 20,000 / 2,621,440 (recipe R, 2026-10-01; was 93,367 / 23,708,768, from 92 windows per trajectory) | The epoch is a count of optimizer steps (one per batch of 2048 windows drawn with replacement) plus a fixed validation set, not a function of the dataset size. Stage 1's values reproduce the old epoch on the 4-file build (N_train 2,078,448 and N_val 257,704 windows): floor(N_train x 32 / 2048) and N_val x 32; the 32 windows per trajectory behind them are inherited with no recorded rationale. In optimizer steps: stage 1 LR warmup 30 epochs = 974,250, plateau patience 15 = 487,125, ES patience 30 = 974,250, 400 epochs = 12,990,000. Stage 2 (recipe R, author 2026-10-01) is a fixed step budget of 20,000 steps per epoch: warmup 1 = 20,000, plateau patience 5 = 100,000, ES patience 12 = 240,000, curriculum ramp 16 = 320,000, 120 epochs = 2,400,000. Its validation set is 1,280 full batches (no ragged last batch) and gives each of the 257,704 validation trajectories 10 or 11 even-stride starts: 11.3% of the 23,193,360 enumerated windows. Its val_loss is not comparable with full-set values, and the selection noise this adds is unmeasured. When raw data is added these counts stay fixed unless edited; if the stage-2 epoch length changes, re-derive every epoch-counted knob from its step target. | inherited (32); stage 2: recipe R, structure-derived, untested |
| `precision.allow_tf32` | true (since 2026-09-30) | TF32 matmuls (cuBLAS through `set_float32_matmul_precision("high")`, cuDNN mirrored), as before. `runtime.deterministic=true` forces TF32 off whatever this key says, the coupling the code always had (deterministic runs are the full-fp32 reproducibility mode). The effective setting is logged at startup and recorded in provenance.jsonl. | inherited |
| **stage-2 model block = stage-1 model block** | 10×2048 (fourier_dt OFF) | Fixes the historical inconsistency: the now-removed `configs/stage2_dt_bands/` declared 8×1536 but strict-load 10×2048 checkpoints from wrong paths — would hard-fail at startup. | high (correctness fix) [rubric: correctness fix, not evidence] |
| `training.rollout_steps` (stage 2) | 10 | **E3 K_MAX sweep** (detached; ranked on the geometric, growing-dt metric): inverted-U — K=1: 6.5e-3–1.9e-2, K=5: 4.4e-3, K=10: 2.4e-3 (±3%, 3 seeds), K=20: 1.2–3.3e-2 + one 1.0-dex fixed-dt blow-up. List et al. (arXiv v1 and v2): detached training of a Kuramoto–Sivashinsky correction CNN diverged beyond ~6 unrolled steps (v2 relates this to t_c/2); our detached E3 optimum among the tested values was K=10, so only the inverted-U shape transfers, not the number (corrected 2026-10-01). K=10 is inside every published envelope (GraphCast 12 max). Training horizon ≪ deployment horizon is universal practice. | medium-high |
| `curriculum` 2→10, linear, 16 epochs (recipe R, 2026-10-01; was 30) | enabled | All multi-step SOTA use staged/ramped horizons (GraphCast +1/1k-updates 1→12; Stormer 1→4→8; NeuralGCM "critical"); sandbox used the equivalent ramp in every successful run. Under recipe R, K rises by 1 every 2 epochs (40,000 steps per rollout length, 320,000 in all, against 2,801,010 for the 30 epochs of 93,367 steps before); the ramp length in steps is untested. | high (literature + used throughout locally) [rubric: below high, never ablated or scale-checked; spot-check candidate] |
| `autoregressive_training.skip_steps` | 1 | Brandstetter 2022 canonical pushforward (unroll 2, backprop last; "more stable"); composes with BPTT. Sandbox used skip 2 at K≥4 successfully; 1 is the literature value and wastes fewer target steps. | medium [rubric: low, skip 1 was not run in the sandbox (inferred); spot-check candidate] |
| `autoregressive_training.detach_between_steps` | **false (full BPTT)** | **E4** BPTT γ=0.9 (2 seeds) vs E3 detached γ=0.9 (3 seeds), compared on both metrics: local evidence mixed and unresolved (corrected 2026-10-01; the earlier text called it a tie within noise). Geometric (growing dt): 2.6–3.0e-3 vs 2.4e-3, both BPTT seeds worse by 8–25%, unresolved at n=2 because BPTT's own spread is 15%. Fixed-dt: 0.30–0.37 vs 0.37–0.55, BPTT better. With 2–3 seeds per arm neither direction is resolved (exact one-sided p ≥ 0.10). BPTT was chosen on the literature prior: GraphCast and NeuralGCM state BPTT; Keisler, FourCastNet, Stormer and ACE2 train multi-step losses without stating gradient flow; among these sources only List et al. (arXiv v2) compares BPTT with detached unrolling (citations below); Kelp 2020 compares 24-step recurrent training with one-step training. Memory: measured GPU peak 5.18 GB on one RTX 6000 Ada (48 GB) at K=10 BPTT, batch 2048, float32 (audit efficiency study); with the production system.device="cpu" the preloaded train+validation set (5,204 B per processed window; 12.2 GB for the 4 local files) sits in host RAM, not on the GPU. | medium (local evidence mixed/unresolved; high literature prior) |
| `autoregressive_training.loss_discount_gamma` | 1.0 (with BPTT) | **E3+E4 interaction** (both ranked on the geometric, growing-dt metric): under BPTT, γ=1.0 gave 3.3e-3 (one reported run; seed count and skip value unverified, EXPERIMENTS.md not on daw) vs γ=0.9 2.6–3.0e-3, 10–27% worse, unresolved at n=1; γ=1.0 was chosen to match SOTA practice, not on local evidence (corrected 2026-10-01; was "≈ ... within noise"). **Under DETACH, γ<1 is required**: uniform was 4–14× worse in both seeds (1.1e-2, 3.3e-2 vs 2.4e-3 at γ=0.9). Hence fallback recipe pins γ=0.9. The code does not enforce it (detach with γ=1.0 trains; a guard is pending the author, open items). | medium-high |
| `backward_per_step` | false | Only meaningful in the detached path; BPTT does one backward, so `backward_per_step=true` with `detach_between_steps=false` is rejected (since 2026-09-30). Fallback recipe sets true (memory). `detach=true` with `backward_per_step=false` gives the fallback's gradients but holds all K−skip step graphs until one backward (9 at K=10, skip 1, against 1 for the per-step fallback: BPTT's activation memory); rejecting that pair too is the author's call. | n/a |
| stage-2 `lr` | 3e-5 (= stage-1 peak / 10) | **E5**: staged pretrain→fine-tune at LR peak/10 gave the best fixed-dt result of the whole campaign (0.256 vs 0.379 from-scratch); LR peak/3 (1e-4) doubled geometric error with no fixed-dt gain. Literature: AR fine-tune LR is 5–3000× below pretrain peak (FourCastNet /5, Keisler /10 per stage, Stormer /100–/1000, GraphCast /3000); /10 is at the aggressive-but-precedented end, and the old template's 2e-5 is also fine. Ranked on the fixed-dt metric. Schedule (added 2026-10-01, audit R3-011): the only surviving record of E5's schedule (the sibling repo's config_stage2.jsonc:17 and :124-125, written 2026-07-29; EXPERIMENTS.md not on daw) says the LR was held flat and not re-warmed, while stage 2 re-warms from 0 and then decays on plateaus. Decided 2026-10-01 (author; open items 8 and 9): recipe R keeps that schedule (warmup 1 epoch = 20,000 steps, plateau patience 5, min_lr 1e-8, which is never reached: torch skips a cut smaller than its eps 1e-8, so from a 3e-5 peak the floor is 3e-5 / 2^11 = 1.46e-8), and the peak is capped at the handoff at min(3e-5, the LR saved in the loaded stage-1 checkpoint) (src/main.py; provenance.jsonl `handoff_peak_lr`). The flat-LR option (warmup 0, min_lr = lr) is not adopted. The schedule is untested. | medium-high |
| stage-2 `wd` | 1e-5 (= stage 1; old template's 0.005 dropped) | No SOTA fine-tune cranks WD ×500 at the phase boundary; the 0.005 value was untested (stage 2 never ran). "×500" is the ratio of the config values; per step, 0.005 would have decayed 33× more than stage 1 at the old template's LR of 2e-5 (50× at 3e-5). Moot in float32 (corrected 2026-10-01; was "keep optimizer geometry constant except LR"): AdamW decays by lr·wd per step, 3e-10 here, below float32's resolution, so wd = 1e-5 is exactly inactive in both stages (stage-1 row; choice open). | medium (literature/consistency) [rubric: low, literature/consistency only] |
| stage-2 `lambda_z_mse` 0.25, EMA 0.999; early-stop 12, plateau warmup 1 | lambda_z_mse and EMA: template values; early stopping and warmup: recipe R (2026-10-01; the template's were 20 / 5) | lambda_z_mse and EMA are inherited from the stage-2 template (stage2_dt_bands at 0bcdffc); not locally discriminated. The early-stopping patience and the warmup are recipe R's step-derived values (dataset and stage-2 lr rows). EMA re-warms automatically at stage start (fresh callback state); ACE evidence says EMA helps long-rollout metrics up to 15%. The re-warm contradicts the flat E5 record (stage-2 lr row); the author kept it. lambda_z_mse (added 2026-10-01, audit R2-032): the z-MSE term is a small share of the loss, 0.34% at λ 0.1 and 0.86% at 0.25 on 20,000 real one-step validation windows, up to ~3.6% on 99-step rollouts at 0.25, and ≤ 0.4% of the gradient norm, so the objective is effectively per-species log10-MAE with a weak z-space MSE term; for the paper, quote the share from a production run's logged val_loss_log10_mae / val_loss_z_mse. | inherited [no spot-check; waiver pending the author] |
| validation compile (`runtime.torch_compile.compile_open_loop_unroll` removed 2026-10-01) | none | Validation reuses the compiled forward step; Dynamo compiles a separate no-grad graph for it plus a graph for the ragged final validation batch under dynamic=false, within the recompile limit of 8: stage 1 forward step 3 variants, stage 2 4 (ragged-validation compile gate, 2026-10-01). Under recipe R the stage-2 validation set (2,621,440 = 1,280 x 2048 windows) has no ragged last batch, so production stage 2 compiles no ragged-batch graph (inferred; the smoke keeps a ragged set and still exercises it). | n/a (bitwise-identical) |
| `runtime.torch_compile.mode` | stage-1 `reduce-overhead` (REVISED 2026-09-30, was `default`) / stage-2 `default` | Stage 1 replays its compiled training step and validation rollouts as CUDA graphs, which cuts host-side launch work: on a CPU-bound proxy (10x256 trunk, logging fixes applied) the stage-1 CPU time per step fell from 3.85 to 2.82 ms (audit efficiency study). On daw's RTX 6000 Ada stage 1 is GPU-bound and wall time did not change (-0.9%, inside noise); a gain is expected only where stage 1 is CPU-bound, e.g. on the GH200 (not measured). Numerics are identical: bitwise 30-step fingerprints (live/EMA weights, every loss, val_loss), and a 3-epoch run that replays every training and validation graph (ragged last batch included, after the EMA weight swap) matches `default` bit for bit (measured 2026-09-30 with the whole-unroll validation graph; the 2026-10-01 forward-step validation is bitwise against it in the 30-step fingerprint). Compile variants with a ragged last validation batch: forward step 3 (no-grad validation incl. the sanity batch, training, ragged validation batch; since 2026-10-01 validation runs on the compiled forward step, before that forward step 1 plus whole-unroll validation rollout 2), 3 CUDA-graph recordings (the recompile limit is 8). Under CUDA graphs the next step's replay reuses the previous step's `.grad` memory, so nothing may read gradients across steps; nothing does (`runtime.log_grad_norm` reads the gradients in both stages, but only within the step, before the optimizer step; since 2026-09-30). Stage 2 keeps `default`: it is GPU-bound (no expected gain), and its no-grad pushforward output feeds the grad-mode step, which the study expects to conflict with CUDA-graph output reuse (inferred, not run). | n/a (throughput only; bitwise-identical) |
| `preprocessing.num_workers` | 16 (REVISED 2026-09-30, was 64) | Preprocessing runs on daw, where one job uses at most 16 cores; the fan-out is also capped at the number of raw files and of cores the process may run on. Outputs do not depend on the worker count (per-file RNG and shard names): 1 and 4 workers gave identical arrays and normalization.json on the synthetic set (audit T1 check). Supersedes the 2026-07-16 "num_workers 64 for parity" note. | n/a (throughput only) |
| `training.early_stopping.min_delta` with ES `patience`, plateau `patience` / `threshold` | min_delta 0.0 (REVISED 2026-09-30, was 1e-5); ES patience 30 / 12; plateau patience 15 / 5 (stage 2: recipe R, 2026-10-01; were 20 / 10), threshold 1e-3 / 1e-4 relative (`threshold_mode` "rel", a named constant in trainer.py) | ES and the plateau observe the same epochs: both ignore the LR warmup and the curriculum ramp and start counting at the first epoch after both (warmup-era epochs stay eligible for best.ckpt). With min_delta 0 every plateau improvement is also a new ES best, and plateau patience + 1 < ES patience (16 < 30, 6 < 12), so once counting has started ES cannot stop before an LR halving. Before, stage 1 could stop at peak LR with zero LR cuts (audit harness, constructed curves with the best inside the warmup: stopped at epochs 35 and 54; with T1 neither stops). Under steady slow improvement `max_epochs` (400 / 120) becomes the binding budget and the relative thresholds decide whether slow progress cuts the LR; tuning them is the author's. The alternative (both monitors keep the warmup-era best, R3-007 option A) was rejected by the author on 2026-10-01 (open item 10). Checked with the real classes on CPU (scripted val_loss, 2026-10-01): stage 1 counts from epoch 29 (stall: cut 45, stop 59); stage 2 under recipe R from epoch 16 (stall: cuts 22 and 28, stop 28; slow steady progress: 11 cuts, the last at epoch 82). A pre-fix run cannot be resumed with these values (the resume config check refuses changed keys). | inherited (min_delta and the shared window: audit) |
| stage-2 `val_loss` definition (added 2026-10-01, audit R1-034) | K = rollout_steps, steps ≥ skip_steps, uniform | Stage-2 val_loss (hence best.ckpt, early stopping and the plateau) averages the open-loop steps after the skip (steps 1..K−1 at skip 1): the first step from a true state is neither trained nor directly monitored. It is comparable across γ, not across skip_steps. Stage-1 best.ckpt (the handoff and the paper's one-step baseline) is chosen by one-step val, which the sandbox found to correlate only ρ = 0.41 with rollout error. Changing the selection rule is the author's (open items). | n/a (definition) |
| `runtime.accelerator` / `runtime.devices` | "cuda" / 1 (REVISED 2026-09-30, were "auto") | Explicit single device: "auto" falls back to CPU silently and Lightning's "gpu" alias tries MPS first; a CUDA request with no visible GPU now fails at startup. The launcher picks the GPU with CUDA_VISIBLE_DEVICES (devices=1 means visible index 0). Multi-GPU/DDP is unsupported. | n/a |
| `system.device` / `training.pin_memory` | "cpu" / true | Every split is preloaded into host RAM (a CUDA preload would put the whole dataset in GPU memory) and batches are pinned for the copy; pin_memory=true with a CUDA preload is rejected. Window indices come from a CPU generator, so the window sequence does not depend on the preload device. Preload budget: a host preload (train + validation, sized from the shard headers) larger than 0.5 × MemAvailable fails at startup instead of being OOM-killed while filling (`_PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE` in src/main.py, not a config key). MemAvailable is a snapshot on a shared host, not a guarantee. The 0.5 is pending the author: at ~785 GB MemAvailable (daw, T1 smoke log) it caps the preload at ~393 GB, below the ~481 GB the audit infers for all 160 raw files, so a full-data run would be refused today. | decision pending (the fraction) |
| `runtime.gradient_clip_val` / `runtime.log_grad_norm` | 1.0 / false | Clip 1.0 is a template value (the v3 reference did not clip; see the Inherited / unjustified table); under AdamW, clipping reweights batches, it does not lower the LR. null is the only "no clipping" value (0 and negatives are rejected). `log_grad_norm` (a required key since 2026-09-30) logs per epoch the mean pre-clip total gradient norm (`grad_norm`) and the fraction of steps where clipping fired (`grad_clip_frac`), in both stages, summed on the GPU. | inherited |
| `precision.*` | float32, amp_mode "true" | The preload is always float32 (`precision.preload_dtype` removed 2026-09-30: bf16 storage moved states by a median 3.1e-3 dex in the audit, above the ~7e-4 target). loss_dtype must be float32 or float64. Under amp_mode "mixed", model_dtype and input_dtype must be float32 and `model.predict_delta` true: a bf16 input, or a whole next state coming out of the bf16 trunk, would carry the rollout state at bf16 resolution. Under autocast the input block and the delta head run in fp32 (fp32 and bf16-true runs are unchanged); bf16-mixed with fused AdamW and clipping trains. bf16-true is kept but lossy: the rollout state is carried in bf16 (on real trajectories a bf16 state cannot absorb 81.6% of the nonzero true per-step changes, audit R4-025). amp_mode "true" means Lightning's `*-true` precision, i.e. no autocast (32-true here; forced for float32/float64 compute; bf16 + "true" is pure bf16); only "mixed" enables autocast, and TF32 is the separate `precision.allow_tf32` key, so the run log's `amp=true` never means AMP or TF32 (added 2026-10-01, audit R2-013). Replacing amp_mode with an explicit Lightning precision string (keeping the dtype cross-checks) is the author's decision. | decision (float32 default) |
| `testing/export.py` `B_MAX` (code constant) | 16384 | The largest batch the export accepts. Inherited with no recorded rationale; kept pending the host envelope (Canoe's per-rank cell count is unknown). The alternative is to drop the cap (`Dim("B", min=1)`, verified bit-identical to eager at B = 65536 by the audit). The .pt2 raises above it (torch.export input constraints); the AOTI package does not check B, its host guard (`_check_step_inputs`) does, and with `AOTI_RUNTIME_CHECK_INPUTS=1` its compiled check would also reject B = 1 (audit). The export gate checks B = 1, 7 and B_MAX and that B_MAX + 1 is refused. | decision pending (the cap) |
| `preprocessing.drop_below` | 1e-30 (= normalization.epsilon) | `drop_below >= epsilon` is required (2026-09-30): every kept raw value is then >= epsilon, so the epsilon floors in the log transforms never change a kept value. | n/a (contract) |
| **rejected: input-noise injection** | not shipped | **E6**: 4/4 noise runs worse, 3/4 catastrophic at fixed-dt (12.4–14.1 dex; σ ∈ {0.01, 0.03} z-space, with and without pushforward, 2 seeds) — *against* the GNS/Stachenfeld prior. No trainer option added. If ever revisited, sweep σ ≤ 1e-3 and re-test. | high (locally decisive) [rubric: below high, 2 seeds, against the literature prior] |
| **rejected: K=20 detached training** | — | E3: consistently worse + one 1-dex blow-up; List 2024 agrees. (K>10 with BPTT untested locally — plausible future direction, not a default.) | medium |
| **not adopted: inference-time renormalization** | off | Sturm 2024: naive projection degrades radicals/trace species (exactly what dominates a log-standardized loss); weighted projection neutral-to-positive. The 12 tracked species are not element-closed in the reference data ("Element closure" below), so hard conservation over them is not a valid target: production must not renormalize the model's outputs or project them onto the simplex, and a stoichiometric output layer (Sturm & Wexler 2022) cannot enforce it either (corrected 2026-10-01; the earlier text proposed tracking conservation drift as an eval metric and called that layer the accuracy-neutral fix). | high (literature) [rubric: below high, literature only] |

## Inherited / unjustified config values

Every value of the two shipped configs that has no per-key evidence row above (structural values
such as paths, species lists, enable flags and the compile settings are left out). "v3 reference" is
the v3 run's `config.resolved.json` (`git show 09748d7:models_old/v3/config.resolved.json`); "library
default" is the torch 2.8 / Lightning 2.6.6 signature default. None of these values was tuned
locally; "inherited" means carried unchanged from the v3 reference, and a value that differs from v3
with no recorded reason is marked so. Stage 1's `training.autoregressive_training` and
`training.curriculum` sub-keys are inert (both blocks are disabled; the one-jump path never reads
them, src/trainer.py) but schema-required.

| Key | Stage 1 | Stage 2 | v3 reference | Library default | Source / status |
|---|---|---|---|---|---|
| `system.seed` | 1234 | 1234 | 1234 | — | inherited; arbitrary seed |
| `preprocessing.seed` | 1234 | 1234 | 1234 | — | inherited; seeds the window draws (split membership uses the code constant `SPLIT_SEED = 1234`, T7) |
| `model.fourier_dt.seed` | 0 | 0 | absent (Fourier-dt added 2026-06-09) | — | new, arbitrary; the frequencies come from their own `torch.Generator` (src/model.py), so they do not change with `system.seed`; inert while fourier_dt is disabled |
| `preprocessing.val_fraction` / `test_fraction` | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | — | inherited |
| `preprocessing.n_steps` | 100 | 100 | 300 | — | changed since v3, no recorded reason; 100 states (99 transitions) per window |
| `preprocessing.shard_size` | 4096 | 4096 | 4096 | — | inherited |
| `preprocessing.max_chunk_attempts_per_source` | 200 | 200 | 200 | — | inherited |
| `preprocessing.t_min` | 0.0 | 0.0 | 0.0 | — | inherited; windows start at max(t_min, first positive raw time) |
| `preprocessing.dt_sampling` | loguniform | loguniform | loguniform | — | inherited (the only implemented value) |
| `normalization.epsilon` | 1e-30 | 1e-30 | 1e-30 | — | inherited |
| `normalization.min_std` | 1e-12 | 1e-12 | 1e-12 | — | inherited |
| `preprocessing.drop_below` | 1e-30 | 1e-30 | 1e-30 | — | inherited value (its bias is measured in the T7 entry; the `>= epsilon` contract has a per-key row) |
| `model.layer_norm_eps` | 1e-5 | 1e-5 | 1e-5 | 1e-5 (`nn.LayerNorm`) | inherited; equals the torch default |
| `model.dropout` | 0.0 | 0.0 | 0.0 | — | inherited (no dropout) |
| `training.loss.lambda_log10_mae` | 1.0 | 1.0 | 1.0 | — | inherited |
| `training.loss.lambda_z_mse` | 0.1 | 0.25 | 0.1 | — | no recorded reason for either value (no commit message gives one; the only related text, a readme.txt passage added at bd1306e and removed at f96385e, both 2026-01-27, gives ranges: 0.01–0.2 "accuracy-first", 0.1–0.5 "early, lower later"). The value went back and forth in git: config.json 0.5 at 23aacde (2026-01-28), 0.25 at a492f39 (2026-01-30), 0.1 from 30e2066 (2026-02-01) to 35f8c58 (2026-02-27), 0.25 again from 55dc644 (2026-03-02); at 43a539b and 09748d7 config.json and every stage-1 dt-band config hold 0.25, while the v3 run's own config.resolved.json (09748d7) records 0.1. The per-stage split first appears at 9738917 (2026-05-14), which set the stage-1 dt-band configs to 0.10 while the stage-2 dt-band configs (added at 0bcdffc, 2026-05-09, at 0.25) kept 0.25; f335fed (2026-06-15) renamed the production pair that carries it (added at 4fd5371) to stage1.json / stage2.json. Whether stage 2 should differ is open (author) |
| `training.optimizer.lr` / `weight_decay` (stage 1) | 3e-4 / 1e-5 | (per-key rows) | 3e-4 / 1e-5 | 1e-3 / 1e-2 | inherited; weight_decay is exactly inactive in float32 (stage-1 per-key row) |
| `training.optimizer.betas` / `eps` | (0.9, 0.999) / 1e-8 | same | same | same (`torch.optim.AdamW`) | inherited; equal to the torch defaults |
| `training.scheduler.warmup_epochs` | 30 | 1 | 30 | — | stage 1 inherited, but v3 warmed a cosine schedule (`cosine_with_warmup`); stage 2 recipe R (2026-10-01; the template value was 5) |
| `training.scheduler.factor` | 0.5 | 0.5 | 0.5 (unused: cosine) | 0.1 | template value; v3 carried the plateau keys but ran cosine, so no v3 run exercised them |
| `training.scheduler.patience` | 15 | 5 | 10 (unused: cosine) | 10 | stage 2 recipe R (2026-10-01; was 10, the torch default); stage 1's 15 has no recorded reason |
| `training.scheduler.threshold` (relative) | 1e-3 | 1e-4 | 1e-4 (unused: cosine) | 1e-4 | stage 2 equals the torch default; stage 1's 1e-3 has no recorded reason |
| `training.scheduler.min_lr` | 1e-7 | 1e-8 | 0.0 (unused: cosine) | 0 | no recorded reason (both are the stage's peak LR / 3000; inferred pattern) |
| `training.ema.decay` | 0.9995 | 0.999 | absent (no EMA in v3) | — | template values (horizons in the lump rows above) |
| `training.early_stopping.patience` | 30 | 12 | absent (no early stopping in v3) | 3 | stage 1 template value; stage 2 recipe R (2026-10-01; the template value was 20); the relation to the plateau patience is in the min_delta row |
| `training.max_epochs` | 400 | 120 | 300 | — | stage 1 changed since v3, no recorded reason (v3's cosine schedule spanned 300); stage 2 recipe R (2026-10-01; was 200) |
| `runtime.checkpointing.save_top_k` | 3 | 3 | 1 | 1 (`ModelCheckpoint`) | changed since v3, no recorded reason; best.ckpt is the top-1 either way |
| `runtime.checkpointing.every_n_epochs` | 1 | 1 | 1 | — | inherited |
| `runtime.gradient_clip_val` | 1.0 | 1.0 | 0.0 | None (no clipping) | not the v3 value: v3's 0.0 means no clipping (Lightning skips clip values <= 0; how the v3-era code passed it is unverified); where 1.0 came from is unrecorded |
| `runtime.accumulate_grad_batches` | 1 | 1 | 1 | 1 | inherited; the Lightning default |

## Documented alternatives (evidence-backed; not shipped as default configs)

1. **Banded layout** — two-band stage1+stage2 over {1e-1–1e1, 1e1–1e3} s. Rejected by
   Gate G2 (band specialists collapsed at deployment-style fixed dt) and **removed** from
   the repo on 2026-06-15 (the former `configs/production/fallback_banded/`). Not a fallback
   under the two-fallback policy (corrected 2026-10-01: the "recover from git history" clause is
   deleted).
2. **Detached stage-2 recipe** (lower memory; the best local geometric point estimate, E3, 3
   seeds; the default rests on the literature prior. Corrected 2026-10-01, was "equally validated
   locally": the sandbox's detached runs used skip 2 at K≥4, while this recipe keeps skip 1 and has
   only been smoke-tested; the seed counts are unverified until EXPERIMENTS.md is recovered). In
   `configs/stage2.json`, under `training.autoregressive_training`, set
   `"detach_between_steps": true`, `"backward_per_step": true` and `"loss_discount_gamma": 0.9`;
   `skip_steps` stays 1 and `training.curriculum` stays 2→10 (`start_steps` 2). Write the nested
   keys: a dotted raw key such as `"autoregressive_training.loss_discount_gamma"` is rejected at
   startup. Do NOT run detached with γ=1.0 (E3: 4–14× worse; not enforced in code, open items).
   A skip-2 variant needs `curriculum.start_steps` ≥ 3 (skip must stay below the epoch-0 horizon),
   and it would be a port of the sandbox recipe, not a replication (author's choice).
3. *Deleted 2026-10-01 (audit R2-019): the deployment-specialized fine-tune (re-preprocess with
   dt_min/dt_max narrowed to ~[2, 20] s, then fine-tune from stage 1). It was not an approved
   fallback: re-preprocessing re-maps the dt input (6.4 s becomes 0.505 instead of the 0.4515
   stage 1 learned) and rebuilds every statistic (weights_only refuses the changed norm_sha256),
   and in the shared processed_dir it would replace the production dataset. If the idea (a second
   stage-2 pass concentrated near dt≈6.4 s) is wanted, it needs the author's approval as a third
   fallback and another route: never re-preprocess, and select the rows of the existing shards
   whose stage-1 dt_norm lies in [0.3253, 0.5753] (2–20 s; no such filter exists in the code).*
4. **Autoencoder trunk** — removed from src on 2026-09-30; recover from git history (tag
   `pre-audit-2026-09-29`). Rationale kept: Fourier-Flow-map was worse + 5.5× seed spread at
   sandbox scale (E1) but small-scale bottleneck behavior may not transfer; documented, not pursued.

## The sibling repo's AR recipe and how Auto-Chem differs (added 2026-10-01, audit R3-013)

Two production ports of the June 2026 Robertson-sandbox AR recipe exist, and neither is the code
that produced the sandbox evidence (not on daw): Auto-Chem's `configs/stage{1,2}.json` (2026-06-15)
and the sibling repo imalsky/Chemulator's `config/config_stage{1,2}.jsonc` (efa5378, 2026-07-29;
unchanged at 071b3e0), which calls itself "the campaign-validated AR recipe". The sibling repo
never mentions Auto-Chem; Auto-Chem cited only the sibling paper until 2026-10-01, when this
document and the paper plan began citing the sibling repo's code and configs. Which port matches
the sandbox runs cannot be checked until EXPERIMENTS.md is recovered; in particular the sandbox's
skip and curriculum handling at K < 3 is unrecorded. Every row is untested at production scale.
Whether to align either side is the author's.

| Setting | Auto-Chem | Sibling (Chemulator) |
|---|---|---|
| rollout gradient | BPTT (`detach_between_steps` false) | detached (`detach_intermediate` true) |
| per-step discount γ | 1.0 | 0.9 |
| pushforward skip | 1; raises if skip ≥ the epoch's K | 2, clamped to K−1 ("always leave ≥ 1 scored step") |
| curriculum | 2→10 over 16 epochs of 20,000 steps (recipe R), K rounded down | 1→10 over 30 epochs, K rounded to nearest |
| stage-2 LR | min(3e-5, the stage-1 checkpoint's LR) after a 1-epoch warmup from 0, then plateau decay (patience 5, min_lr 1e-8) | flat 3e-5 (warmup 0, min_lr = lr) |
| stage-2 handoff | stage-1 best.ckpt (since 2026-09-30) | stage-1 best.ckpt |
| z-space loss weight | 0.1 (stage 1) / 0.25 (stage 2) | 0.0 |
| weight decay | 1e-5 (inactive in float32) | 1e-4 |
| architecture | 10×2048 residual MLP (FlowMapMLP) | latent_linear (512-d latent, 3×4096 encoder and decoder) |
| validation weighting | uniform over steps ≥ skip | γ-weighted, like training |

## Production-scale spot-checks (recommendations pending the author)

None has been run; which to run, and each protocol and decision rule, are the author's (rewritten
2026-10-01; the earlier list compared ~20 stage-1 epochs on one GH200 and the first 10 stage-2
epochs, and called fourier_dt "the single most load-bearing adoption", which it no longer is).
Facts that constrain any design:
- Stage 1 warms the LR up over 30 epochs, so the first ~20 production epochs end at 2/3 of the
  peak LR with the plateau never stepped; a standalone job with max_epochs = 20 and warmup_epochs
  = 30 silently caps the warmup at min(warmup_epochs, max_epochs) = the whole run, which is not
  the production schedule either.
- Stage 2: under the shipped curriculum (2→10, linear, ramp 16 under recipe R, skip 1) training K
  by epoch 0–9 is 2,2,3,3,4,4,5,5,6,6 and first reaches 10 at epoch 16. At K=2 with skip 1 the BPTT+γ=1.0 and
  detached+γ=0.9 arms are the same computation (audit: bitwise-identical weights, EMA weights and
  val_loss on CPU; on GPU up to nondeterministic reductions, inferred), and the 1-epoch warmup
  covers epoch 0. A first-10-epochs A/B therefore cannot test the K=10 recipes.
- A resume must continue the same run with the same config (T1 entry), and weights_only restarts
  the curriculum, the LR warmup and the optimizer state at epoch 0, so forking a running stage 2
  into a second recipe after epoch 3 has no code path today.

Candidate checks:
- Stage 1, each arm under the production max_epochs for at least the warmup plus a tail at peak
  LR where the plateau can act (e.g. ≥ 40 epochs), compared after the warmup; or a disclosed
  standalone short job with a scaled warmup (e.g. 30/400 of its length):
  - the primary (residual=true, Fourier off) vs the approved fallback pair (residual=false +
    fourier_dt.enabled=true, flipped together);
  - a production-scale transfer check of the F1/F2 result (residual trunk, Fourier on vs off),
    only if the author still wants it (configs/stage1.json's fourier_dt comment refers to it);
  - fourier_dt.sigma ∈ {0.5, 1, 2} on the fallback trunk, only if the pair check shows
    sensitivity.
- Stage 2, BPTT+γ=1.0 vs detached+γ=0.9: both arms from the stage-1 best.ckpt with the curriculum
  off (K=10 from the first step), compared after the warmup and labelled "BPTT vs detached
  at K=10 without the ramp" (it is not the shipped ramped schedule and cannot continue as the
  production run); a fork after the ramp would need the code path above. Pre-register the decision
  rule (a seed-spread rule needs ≥ 2 seeds per arm at scale) and score on the fixed validation
  windows plus a fixed-dt validation metric. If neither is affordable, record the E4 comparison as
  untested at scale.
- Rows labelled low/inherited without another check (audit R4-004): curriculum on/off or ramp
  length; skip 1 vs 2 (needs curriculum.start_steps ≥ 3, so it is confounded with the curriculum
  start); the stage-1 and stage-2 template rows (scheduler, EMA, early stopping). Each is a
  candidate check or an explicit waiver. The other rows labelled inherited (the trunk, whose study
  is deferred; the epoch counts; TF32; gradient clipping; the early-stopping and plateau patience
  row) have no check listed: waiver pending the author.
Every run uses one daw GPU through run_daw.sh.

## Verification performed (2026-06-09; historical)

*Historical record of the 2026-06-09 working tree (before the first commit holding this block,
4fd5371); it was not re-verified when src/ changed later. "all 7 configs" and "fallback_banded"
refer to the pre-2026-06-15 layout. The current record is the next section (marked 2026-10-01).*

- **End-to-end smoke** (`testing/run_local_smoke.sh`): synthetic raw → preprocessing → stage-1
  (2 epochs) → stage-2 default (BPTT) AND stage-2 fallback (detached+γ=0.9), with the
  weights_only EMA handoff. PASSED on this machine. (Since 2026-09-30 the smoke is the integration
  gate and asserts its outcomes: compile on, extension and refused re-run, best.ckpt handoff, the
  Fourier-dt fallback pair, the gated export; spec 4.4.4.)
- **Bugs found and fixed during verification:**
  1. *Latent stage-2 crash* (pre-existing): Lightning forbids `Trainer(gradient_clip_val)`
     with manual optimization — every AR run would have died at fit start. The AR path now
     clips in-module (`norm`, same value) and the Trainer arg is withheld in AR mode.
     One-jump mode still clips via the Trainer.
  2. *Export shipped the wrong weights* (pre-existing): `testing/export.py` ignored
     `ema_state_dict`, so with EMA enabled the deployed .pt2 would contain live weights
     while checkpoint selection/val_loss used EMA weights. Export now prefers
     `ema_state_dict`, mirroring `src/main.py`.
- **Adversarial code review** (two independent reviewers over the diff): default-off
  behavior verified *bitwise* against HEAD (state dicts, forward outputs, gradients);
  γ math consistent across detached/accumulated/BPTT paths; torch.export verified with
  dynamic batch at fp32 and bf16 (a shape check only: the output batch size; bf16 numerical
  fidelity was never verified. A 2026-09 audit measured the bf16 export at a median 6.2e-3 dex
  error against 3.3e-7 for fp32, and the bf16 export option was removed on 2026-09-30).
  Review findings addressed: fallback_banded path depth fixed (+ resolution verified);
  `fourier_dt` block without `enabled` now raises; `loss_discount_gamma` outside AR mode now
  raises; Fourier projection computed in at least float32 internally (bf16 upcast, float64 kept;
  bf16-config fidelity). Known
  non-blocking notes: val_loss is intentionally uniform-weighted even when training γ<1
  (cross-config comparability — commented in trainer.py); bf16-mixed + fused optimizer + clipping failed at
  the first step in both stages (Lightning's GradScaler-only fused-optimizer guard). Since 2026-09-30 (audit
  T1) the module clips with `clip_grad_norm_` itself and that combination trains in both stages (30-step
  runs bitwise equal to the efficiency study's patched runs); production configs are fp32-true.
- ruff/vulture clean on all changed files; all 7 configs pass the strict schema;
  legacy configs unaffected. (The ruff rule set was not recorded then; since 2026-09-30 it is pinned
  in pyproject.toml `[tool.ruff.lint]`: E4, E7, E9, F, with ruff 0.16.9 from requirements.lock.
  "7" = the configs present on 2026-06-09, config.json plus 6 under configs/production/; since
  2026-06-15 only configs/stage1.json and stage2.json remain, and both pass the strict schema,
  re-checked 2026-09-29 and 2026-10-01.)

## Re-verification 2026-09-29 to 2026-10-01 (audit branch `audit-2026-09-29`, from b4760b8)

Every check ran on daw in the uv venv (Python 3.12, torch 2.8.0+cu126, Lightning 2.6.6), with 16
pinned CPU threads and one RTX 6000 Ada. Tools: ruff 0.16.9, pyflakes, vulture.
- Baseline b4760b8 (2026-09-29): the smoke passed with compile off, on CUDA, with a
  7,180-parameter [64, 64] model on synthetic data; ruff `--select E4,E7,E9,F`, pyflakes and
  `vulture --min-confidence 100` clean; both production configs pass the strict schema. Not
  re-checked then: the bitwise default-off comparison against pre-June code, the γ-math
  consistency across the detached/accumulated/BPTT paths, torch.export/AOTI at fp32 and bf16 with
  dynamic batch, and the compile-on path.
- Audit commits: T6 27217bc; T3 3177adb; T1 78d7930..93c1e83; T7 3fbd779..af23642; T4 e9592b8;
  T5 9a14210..e2d755f; T2 81b515a..3a682c3; T9 67a7f7b..1b11fea; T8 (docs and comments). Every
  theme that changed training, preprocessing or export code or a config (all but T2, which
  changed the launcher, the smoke and the docs) re-ran a deterministic 30-step training
  fingerprint of both stages (live and EMA weights, every training loss, val_loss) on synthetic
  production-shaped data against the previous reference: bitwise except where the commit
  intended a change (T1: epoch-keyed training stream, sanity batch, EMA buffers; T7:
  deterministic validation and redrawn windows; T9: the config-based warmup, bitwise under
  overrides that make the old and new warmup agree). Every theme ran the smoke (compile on and
  asserting its outcomes since T2) and pyflakes, ruff `F,E9,B,PLE` and `vulture
  --min-confidence 100` on its touched files.
- Theme checks: compile-variant gates with a ragged last validation batch (T3, T1, T7, T9; today
  stage 1 forward_step 3 variants, stage 2 4, limit 8); resume equal to an uninterrupted run (T1
  for the training stream, T7 including every val_loss); preprocessing equal array by array on
  synthetic and real-data subsets (T6, T1, T5, T9) and one real 4-file build (T7); export
  equivalence and the export gate on CPU and CUDA (T6, T1, T7, T5); bf16-mixed with fused AdamW
  and clipping trains (T1).
- Not run on daw: AOTI compile and load (torch 2.8 extracts packages into the shared system temp
  dir; the Python flow ran with compile and load stubbed); any full or real-data training run;
  mini-chem timing.
Future entries record the commit hash, the tool versions and the rule sets.

## Element closure of the 12 tracked species (added 2026-10-01, audit R1-014, R4-007)

The 12 species sum to 1 at t = 0, but the sum falls below 0.99 at some t > 0 in 10.6% of kept
groups (19.8% at 300–600 K); the t_end deficit reaches p99 0.092 and max 0.94; element ratios drift
(C/H p90 21%, N/H 39%); groups rejected by drop_below close worse (the sum falls below 0.99 in
14.7% of them vs 10.6% of kept groups). The raw files cannot tell untracked species from solver
non-conservation (the generator config is not on daw). Hence:
- production must not renormalize the model's outputs or project them onto the simplex, and a
  stoichiometric output layer cannot enforce 12-species element conservation, because the
  reference data violate it;
- the per-species log-log interpolant under-counts the C, O and N totals between raw outputs
  (weighted AM-GM): against the straight-line blend of the raw totals, native-midpoint p99 is
  about 0.9%, the fixed-dt mean 3–4e-4, and about 4–5% of K=100 windows contain a step above 1%;
  against the truth at 2× spacing, 93–97% of the error mass is negative. Pooled per-species dex
  errors are nearly unbiased (CO and H slightly negative);
- for any future element metric (recommendation, nothing computes one today): score the model's
  element totals against the truth's at the same step ("element-budget error vs truth": E = y @ Aᵀ
  with a C/H/O/N stoichiometry table as a named constant, per-step p50/p90 of the signed
  (E_pred − E_true)/E_true per element, plus the signed per-species median log10(y_pred/y_true)),
  never as conservation drift; build the truth's totals by log-log interpolation of each raw element
  total with the interpolant's own (i0, i1, w), E_ref = E[i0]^(1−w) · E[i1]^w, never as the sum of
  interpolated species. A grid t0 + k·dt cannot land on the raw grid for K ≥ 100, so a
  raw-time-aligned fixed-dt comparison is not possible.

## Label accuracy: the interpolation floor (added 2026-10-01, audit R1-016; estimates)

Every training and validation label is a log-log interpolant between raw outputs. The raw grid is
t = 0 plus 99 positive save times from 1e-3 to 1e8 s, 0.1122 dex apart (all 4 raw files); 90% of
training transitions have no raw output strictly inside them. Estimated floors (inferred,
Richardson extrapolation from 2h and 4h subgrids): per-step increments about 2–5e-4 dex mean;
label values about 1e-3 dex mean (worst interval position p90 about 1.25e-3, p99 about 0.022 dex;
far worse below 1e-25, leave-one-out p99 0.54 dex). So:
- run-to-run one-jump val_loss differences below about 5e-4 near the former ~7e-4 gate are not accuracy
  differences;
- stage-2 K=10 val_loss carries about 0.5–1.5e-3 dex of label noise (inferred);
- no absolute-accuracy or per-species p99 claim below about 0.02 dex can be made against an
  interpolated reference, including the planned fixed-dt evaluation set;
- val_loss is comparable only while the raw cadence and the interpolation are unchanged: check the
  time grid of new raw files (preprocessing_summary.json records it per file).
Trace levels enter log10-MAE at full weight (epsilon = drop_below = 1e-30); points below 1e-20 carry
about 14% of the interpolation error. Constant raw tails (open items) biased these estimates low
(leave-one-out p90 at 1e7 s: 1.4e-3 dex with them, 3.8e-3 without). Measuring the true floor needs
dense-output VULCAN reruns for a few (T, P) (recommendation).

## Validation protocol (acceptance gates for the trained models)

The three gates are the author's (2026-06-09). Audit corrections and open points are marked
(2026-10-01). The runs happen on daw through run_daw.sh (the HPC route is retired).

1. Stage-1: **skill against persistence (PROVISIONAL, author 2026-10-01; revisit after the first
   trained model)**. Pass if, overall, the model's loss is at most 0.5 × persistence's, and it is
   below persistence's (ratio < 1) in every window-start decade, especially from 1e3 s on. The loss is
   the run's val_loss definition; persistence is y_next = y. Reported by
   `testing/eval_rollout_error.py` as pass/fail lines (constants `GATE1_MAX_LOSS_RATIO`,
   `GATE1_MAX_DECADE_RATIO`, `GATE1_LATE_START_S`), never enforced by training or by the exit
   status. Training logs the same comparison per epoch as diagnostics (`val_persistence_loss`,
   `val_skill` in metrics.csv and init_val.json). Replaces the earlier gate "val one-jump loss ≤ v3
   reference (~7e-4)", whose **reference is not comparable (spec 6.7)**: ~7e-4 is v3's first
   crossing (epoch 45); v3's recorded minimum is 4.03e-4 (epoch 207), on other data and validation
   windows ("Where the evidence is"); persistence scores 0.0112 on the stage-1 validation
   definition, about 16× that threshold (open item 2). The interpolation floor limits what the gate
   resolves ("Label accuracy").
2. Stage-2 (the deliverable's real test): held-out test-split **open-loop rollouts at
   fixed dt = 6.4 s for K ≥ 100**, plus dt = 0.5 s and dt = 100 s; require 100% finite
   trajectories, error growth that flattens after the transient (the sandbox shape:
   front-loaded, then plateau — no super-linear growth past ~step 20), and 90th-pct
   per-species log10 error well below the paper's one-shot-trained 56.6% @ 99 steps
   (target < 10%). Track species-wise bias drift, especially any species lacking strong
   chemical sinks (Kelp 2022's HNO3 lesson), and the model's element budget against the truth's
   at the same step as metrics (corrected 2026-10-01; was "element-conservation drift": the truth's
   own totals drift, see "Element closure"). Open points (audit; defining the target is the
   author's, open items): the target mixes dex with percent (a 90th-percentile error of 10%
   fractional is 0.0414 dex = +10% / −9.1%, while "< 10%" read as "< 0.1 dex" allows +25.9% /
   −20.6%; the one-shot 56.6% is 0.195 dex for over-prediction and 0.362 dex for
   under-prediction); no fixed-dt evaluation set with K ≥ 100 exists yet (the processed shards
   give K ≤ 98); and a horizon of about 100 steps (640 s at 6.4 s) cannot see slow per-step drift
   (on the test split 75.0% of kept groups still change by > 1e-3 dex per raw interval, in a
   species ≥ 1e-20, after 640 s), so "long-horizon" claims hold only to the tested horizon.
3. Cross-check vs mini-chem in the Canoe harness (`Canoe-Chem`, dt≈6.4 s) before swap-in.
   Units at the gate boundary (added 2026-10-01, audit R4-008): the emulator's P input is dyn/cm^2
   (barye), as trained (raw P 1 to 1e10); mini-chem's P_in is Pa (upstream mini_ch_i_dlsode.f90:38,
   `P_cgs = P_in * 10.0_dp`; the mini-chem source is not on daw). Convert at each code's input from the harness's native unit,
   P_Pa = BARYE_TO_PA * P_barye with BARYE_TO_PA = 0.1, a named constant in whatever script
   implements the gate. Neither side catches a factor-10 slip: mini-chem has no P check, and the
   export's out-of-range NaN rule would flag only about 10% of Pa-for-barye inputs (estimate,
assuming log10 P uniform over the trained 1 to 1e10 barye; the real P distribution was not used). The
   sibling repo shipped this error once (fixed in its eb122d0). Which side converts is not stated
   until Canoe-Chem's native pressure unit is checked. Species inputs: the host's mixing ratios
   (VULCAN normalizes over all species, He included) are not the training convention (spec 4.3);
   pass-through vs renormalizing the 12 species is an OPEN decision (open items).

## Open items for the author (audit 2026-09-29; recorded 2026-10-01)

Each item is a decision the audit left to the author, with the audit's numbers (measured on the 4
local raw files unless noted; "inferred" and "estimate" mark what was not measured) and, where one
exists, the audit's recommendation. Items already recorded in a row above are pointed to.

**Evaluation, gates and the paper headline**
1. Stage-2 target (R2-021, R2-027). **Decided 2026-10-01 (author): a reported target, not a hard
   gate** (spec 6.3): per-species p90 in dex (pooled value also reported) on the validation split
   during development and on the test split once at the end, at a stated step and dt, with the %
   equivalent, against the stage-1 export run through the same script (not the sibling's pooled
   56.6%). eval_rollout_error.py states the step (from its start index) and the dt with every
   headline row. Still open: the threshold (numbers under gate 2) and a fixed-dt set (item 5);
   today the script scores random-dt windows (each window's own constant dt) with K ≤ 98.
2. Gate 1 (R1-013). Persistence (y_next = y) scores 0.0112 on the stage-1 validation definition and
   0.0308 on the stage-2 masked K=10 definition (rebuilt real chunks, approximate statistics); 71% of
   its stage-1 loss comes from the top 1% of windows, and its per-window loss falls from 0.023
   (t_start < 0.1 s) to 1.8e-5 (> 1e5 s), so val_loss mainly measures early transients.
   **Decided 2026-10-01 (author), implemented, PROVISIONAL**: val_persistence_loss and val_skill
   (ratio of epoch means) are logged as diagnostics; eval_rollout_error.py adds persistence curves
   and a loss-ratio and skill table by window-start decade, and reports gate 1 (gate 1 under
   "Validation protocol": ratio ≤ 0.5 overall, < 1 in every decade). Revisit the thresholds after
   the first trained model. Unverified audit point (R9-012): persistence may already meet a
   stage-2 target for late starts; the eval's persistence rows now show it.
3. Selection metric (R1-034; row "stage-2 val_loss definition"). Recommendation: one
   behaviour-neutral diagnostic val_loss over all K steps when skip > 0. If rollout re-ranking is
   wanted: a second, monitor-free ModelCheckpoint every N epochs (about 0.61 GB per full
   checkpoint; never save_weights_only, which drops the EMA) and a re-rank script on validation
   shards, applied to both stages.
4. Long horizons and steady states (R3-002). On the test split, 82.4% / 75.0% / 66.7% of kept groups
   still change by > 1e-3 dex per raw interval (in a species ≥ 1e-20) after 50 s / 640 s / 1e4 s;
   the median last change is at 2.68e6 s; 48% end in converged tails; a steady bias of 1e-5
   dex/step shows only about 1e-3 dex by step 100. Recommendation: (a) a long-horizon and (b) a
   fixed-point (start inside a converged tail) diagnostic as modes of the planned fixed-dt
   generator, gated on validation, with equal step budgets, scored at raw output times (cost from
   75.8 MFLOP per cell-step: (a) 3.6e16 FLOP, (b) 2.3e16 FLOP, estimates; ask before running).
   Until then, qualify "Stable Long-Horizon" and "GCM-deployable" to the tested horizon. A
   drift-rate bound would be a new acceptance threshold (spec 6.3).
5. Fixed-dt evaluation generator (R1-043): not implemented. It must start at k ≥ 1 (T7 entry);
   whether it also gets a changing-dt arm is item 21.
6. Headline confounds (R1-009). The stage-1 vs stage-2 comparison also changes lambda_z_mse
   0.1 → 0.25, adds up to 120 epochs (recipe R) with a fresh 1-epoch warmup to min(3e-5, the stage-1
   checkpoint's LR), and uses one seed (1234).
   Recommendation: a step-matched one-step continuation control generated from stage2.json
   (curriculum start = end = 1, skip_steps 0, AR on with rollout_steps 10 so validation stays
   K=10; everything else identical), matching optimizer steps and reporting the ~7.1× per-step
   cost ratio; one selection rule for all arms; a paired, source-level bootstrap CI; ≥ 2 stage-2
   seeds varying only system.seed (each extra stage-2 run is up to about 2.3 GPU-days under recipe R,
   estimate). The
   handoff anchor is now recorded (init_val.json, T7).
7. ROLLOUT_START_INDEX (R3-029): eval_rollout_error.py starts every rollout at index 1 with no
   recorded rationale (training windows start at index 0); on the smoke model the p90 at step 10
   was 181.8% at start 1 and 794.3% at start 0. Setting it to 0 is the author's call.

**Training recipe**
8. Stage-2 LR schedule (R3-011; stage-2 lr row). **Decided 2026-10-01 (author)**: recipe R keeps
   the warmup and plateau schedule (warmup 1 epoch, patience 5, min_lr 1e-8, threshold 1e-4,
   factor 0.5); the flat 3e-5 option (warmup_epochs 0, min_lr 3e-5) is not adopted. The schedule
   is untested; the first stage-1 learning curve is the evidence to revisit it with.
9. Stage-2 LR at the handoff (R2-015). Stage 1 halves its LR on plateaus (factor 0.5, patience 15,
   min_lr 1e-7), so after 4 halvings it sits at 1.875e-5, below stage 2's 3e-5. **Decided
   2026-10-01 (author), implemented**: a weights_only start caps its peak at min(3e-5, the LR saved
   in the loaded checkpoint's optimizer state); src/main.py logs it and records it in
   provenance.jsonl (`handoff_peak_lr`), and a resume keeps it (scheduler state). Each halving
   below 3e-5 removes one cut on the way to the floor (from 1.875e-5: 10 cuts, so a full anneal
   needs at least 77 epochs instead of 83 under recipe R; checked with the real classes).
10. Early stopping (R3-007 option A; min_delta row). **Decided 2026-10-01 (author)**: option A
    (both monitors keep the warmup-era best; audit harness: LR cut at epoch 44, stop at epoch 58)
    is rejected; the T1 rule stays (both start fresh at the first epoch after the warmup and the
    ramp).
11. Weight decay (R4-006; stage-1 row): 1e-5 is exactly inactive in float32. Options: (a) keep 1e-5
    labelled "inactive in float32"; (b) set 0.0 and delete the parameter-group builder and the
    exclude_norm_and_bias key (numerics unchanged; resuming an older checkpoint then fails on the
    param-group count); (c) a value that registers (> ~1e-4 at lr 3e-4), chosen from lr·wd·steps
    and tested.
12. lambda_z_mse (R2-032; stage-2 template row): σ_s is 3.3–7.0 dex for 11 species and 0.43 dex for
    H2; retraining a small model with λ ∈ {0, 0.1, 0.25, 1} changed the MAE by less than the seed
    spread. Optional: one value for both stages.
13. Ramp-phase window bias (R1-091): windows are sized for K=10 (starts in [0, 89]) while K ramps
    2 → 10 over 16 epochs (320,000 steps under recipe R; ~2.8M before); at K=2, transition positions 91–98 and 0 are never
    targets, and even fully ramped the chunk edges get 1/9 of the weight; measured total variation
    ≤ 0.015. Recommendation: accept and document (small, ramp only); alternative: a per-sample
    offset o ~ U[0, K − k_train].
14. Guards not added (R4-019 = R1-070, R2-033, R2-020): reject detach_between_steps=true with
    loss_discount_gamma 1.0 (recommended, in main.py's early validator; it changes spec 4.2.28,
    where an absent gamma reproduces legacy behaviour); reject detach=true with
    backward_per_step=false (the fallback's gradients at BPTT memory); a create_model guard for
    residual=false without Fourier (the E7 collapse).
15. Single-valued model keys (R4-017): model.dropout, model.layer_norm and model.predict_delta take
    one value in production and in both fallbacks. Option 1: remove the keys (state dicts
    unchanged); option 2 (today): keep them, guarded by the model-block check at the handoff and at
    export.
16. bf16-true (R4-025; precision row): kept per policy but lossy (rollout state in bf16).
    Alternatives: reject compute bf16 with amp_mode "true", or collapse precision to float32 +
    allow_tf32. bf16-mixed is about 47% faster (audit efficiency study) but its accuracy is
    untested; float32 stays the default.

**Data**
17. Frozen raw tails (R2-006): 326,167 of 649,084 kept groups (50.25%) end in bitwise-constant tails
    (1.8% at 300–600 K, 99.9% at 2700–3000 K; 28.5% at log10 P 0–1 vs 68.9% at 9–10); 17.3% of
    transitions and 11.2% of chunks are exact no-ops (62.4% / 35.3% at 2700–3000 K), where
    persistence scores exactly. 95% are converged (last change p50 2.5e-14 dex), but 16,311 groups
    (2.5%) froze mid-evolution (species at 1e-28..1e-19 changing up to 2.19 dex per raw interval:
    N2 12,833, C2H2 1,082, HCN 899, O 680). Recommendation: truncate, not reject (a named
    _FROZEN_TAIL_MAX_PRECHANGE_DEX = 1e-3; keeps ~84% of the affected chunks, loses 0.39% of all
    chunks; 429 groups lose all chunks), then rebuild and retrain; report steady-state rows as a
    separate evaluation stratum. The audit's float32 and leave-one-out numbers were biased by these
    rows.
18. Sub-1e-20 labels (R2-014): the generator's solver tolerances and stop criterion are unrecorded,
    and epsilon = drop_below = 1e-30 is inherited against a measured ~1e-20 solver-control floor.
    Below ~1e-20 the raw series are much rougher: isolated > 0.3 dex excursions at 2.5e-3 per point
    below 1e-25 and 9.6e-4 at 1e-25..1e-20, vs 1.9e-6 at ≥ 1e-5 (lag-1 correlation of second
    differences −0.29 / −0.24 vs +0.13, a noise signature); 53.8% of kept groups and 4.4% of kept
    values sit below 1e-20 (O 15.2%, C2H2 11.2%). Recommendation: recover the VULCAN-0D generator
    config and have the evaluation report error separately outside solver control (thresholds as
    named constants); no loss change.
19. T7 entry items: the drop_below bias (reweighting, a per-cell gate or a composition-aware
    out-of-distribution flag); the window-start weighting (25% of windows start at exactly
    1e-3 s); training never sees t = 0 (a fixed-dt generator must start at k ≥ 1).
20. Split key (R3-015(b)): keying the split on the group name alone (rename-proof) re-splits
    304,702 of 896,000 groups (34.0%), changes N_train/N_val, and forces a second real build,
    re-derived epoch counts and new parity references.
21. dt schedule within a rollout (R4-012): spec 4.3.10 makes variable per-step dt mandatory, but
    every trained, validated and evaluated rollout keeps one dt. Options: (a) give the planned
    fixed-dt generator a separately reported changing-dt arm (piecewise-constant or alternating
    in-range dt; truth from the same log-log interpolation at cumulative times) and keep dt stored
    as [N, T-1]; or (b) reword spec 4.3.10 and README lines 4-6 to "the one-step export accepts any
    in-range dt per call and per row; multi-step rollouts are trained and validated only at
    constant dt within a rollout; rollouts whose dt changes between steps are unvalidated", and
    put the same sentence in the export metadata next to the units.
22. Per-window dt storage (R2-028): dt is one value repeated 99 times per window, 7.61% of shard
    bytes and of the train+validation preload (about 0.9 GB of about 12 GB on the 4 files;
    22–37 GB at the inferred full scale); a per-window [N] buffer saves 7.53%. It waits on item 21:
    keep [N, T-1] if changing-dt windows are planned.
23. Preload budget (R1-023; system.device row): 0.5 × MemAvailable would refuse a full 160-file
    preload today (~393 GB cap vs ~481 GB inferred), and data larger than RAM has no loading path.

**Numerics and deployment**
24. float32 host state (R1-046 with R3-003). Floors: the float32 half-ulp of log10 y is 6e-8 to
    9.5e-7 dex and of linear y 1.3–2.6e-8 dex; the float32 z-add floor (half-ulp(z) × log_std) is
    p50 1.5e-7, max 1.2e-6 dex. At dt = 6.4 s, 46.5% of kept (state, species) pairs change by less
    than the float32 resolution (37.0% excluding constant raw tails); with a perfect delta, 4.1%
    freeze while still relaxing (12.4% at 1e7–1e8 s), missing a median 0.14 dex by 1e8 s; 39.8% of
    the training target deltas are stored as exactly zero; K ≤ 100 evaluations cannot see it; the
    deployment impact is inferred. Options: (1) a partial mitigation, not a standalone fix (R3-003),
    with no retraining or contract change: a FlowMapMLP hook returning the network output before
    the residual add (bitwise identical) and float64 species log10 / z / residual add / 10^x inside
    the export wrapper, float32 I/O kept (export and AOTI verified on CPU). Frozen while relaxing
    at 6.4 s: 5.34% → 1.93% per pair (4–5% at t ≥ 1e6 s); about 5% weighted by deployment steps,
    against about 12% today, so it misses the audit science panel's ~1%-of-deployment-steps
    trigger; useful only as a step toward (2); (2) the complete fix needs the host: a float64 host
    state through float64 I/O or an export returning dlog10 y (a spec 4.3 amendment), measured
    0.002%; sub-cycling to dt = N × 6.4 s is a host-side mitigation. Option (1) alone is therefore
    not a fix: every working option needs a float64 host state and the residual add outside the
    float32 network (option (1)'s hook). Float64 training deltas would need both float32 casts in
    preprocessing removed, at 2× temp-shard disk.
25. Host species convention (R3-008): the export takes mixing ratios in the training convention
    (spec 4.3), while VULCAN normalizes over all species, He included. A real He-bearing column
    (HD209, 69 species) gives a 12-species sum of 0.857 (x_He 0.143) with H2 99.8% of it; that
    combined state is 2.8e-5 of training states, against 27.4% for the He-free equivalent. OPEN:
    pass the host's ratios through, or renormalize the 12 species to sum 1 (which counts the He as
    reactive gas). History: commit 43a539b (2026-03-23) replaced testing/vulcan.py, which
    renormalized, by aoti_export.py's pass-through with no note, so copy neither from old code.
    Recommendation: gate any host coupling on a He-bearing VULCAN-0D box compared against truth
    under both conventions.
26. AOTI on daw (R3-009): torch 2.8 extracts each loaded AOTI package (about 152 MB at production
    size) into a hard-coded directory under the root temp dir whatever TMPDIR says, and a killed
    run leaves it there. Routes: (a) torch ≥ 2.13, whose loader honours TMPDIR (conflicts with the
    torch 2.8.* pin: a whole-project upgrade gated on parity); or (b) stay on 2.8 with a zipfile
    extract into TMPDIR plus a private runner API confined to aoti_export.py. Until one is chosen,
    aoti_export.py is not run on daw.
27. Export details (T7 and T5 entries): B_MAX = 16384 or no cap (per-key row); MATCH_MAX_DLOG10 and
    EXPORT_PARITY_RTOL to re-check on the first trained model; DEPLOY_BATCH_SIZES = (150,)
    unconfirmed for the VULCAN-0D reference and Canoe; `cuda_visible_devices` was removed from the
    metadata (confirm the external VULCAN loader does not read it); aoti_export.py's
    REFERENCE_VULCAN_STATE_PATH and INSTALLED_RAW_MODEL_PATH do not exist on daw (give their
    location or retire the installed-model comparison).
28. Speed (R1-002, R3-012): mini-chem must be rebuilt and timed on daw (its source and per-cell
    driver are not on daw; gfortran is), at the emulator's pinned thread count and with P converted
    to Pa; whether to build the AOTI package with cpp.dynamic_threads is open.

**Repository and evidence**
29. Evidence recovery (R1-010, R2-014): before the old machine is retired, recover EXPERIMENTS.md,
    IMPLEMENTATION_PLAN.md, Robertson_Tests (runs_f*, runs_l*, figures_l1), the mini_chem source
    with bench_dlsode.f90 and bench*.nml, models/v3 (export, config, bench_speedup.csv), models/0,
    Chemulator_ApJ and the VULCAN-0D generator config. Version small text evidence (docs/evidence/
    or an evidence repo); archive binaries and run dirs under a DOI with their location recorded.
30. License and availability (R2-012): the repo has no LICENSE or CITATION.cff; its GitHub copy is
    public (imalsky/Autoregressive-Chemulator, main = b4760b8 when audited), so others have no reuse
    grant (default copyright); the sibling repo is MIT. Recommendation, after the author picks a
    license: LICENSE, pyproject `license` / `license-files`, a CITATION.cff, and a README "Code and
    data availability" section (which raw files back the paper; where the raw files, both best.ckpt
    and the exported .pt2 will be archived). The AAS policy only recommends a license.
31. Process and unverified points: spec 8.4 ("tests are a non-goal") is the author's to rewrite
    (R9-023); whether run_daw.sh should set TORCHINDUCTOR_COMPILE_THREADS=1 (reported, unverified,
    to avoid transient semaphores on the shared RAM disk; R9-032). Further R9 gap-round points
    whose verifiers did not finish are unverified, among them the fullgraph/dynamic compile
    defaults (R9-011), dropping the unused "standard"/"log-standard" normalization methods
    (R9-021), a restricted unpickler for .vul inputs (R9-035), and removing the curriculum-off,
    skip-0 and scheduler-off arms (R9-034).

## Key citations (checked against primary sources 2026-06-09; the List et al. entry was re-checked against arXiv v1 and v2 by the 2026-09 audit and corrected 2026-10-01; the others were not re-checked)

Brandstetter, Worrall & Welling 2022 (ICLR, arXiv:2202.03376) — pushforward trick ·
Lam et al. 2023 GraphCast (Science 382:1416, arXiv:2212.12794) — BPTT, 1→12 curriculum, LR 3e-7 ·
Nguyen et al. 2024 Stormer (NeurIPS, arXiv:2312.03876) — single dt-conditioned model, staged K, LR /100–/1000 ·
Keisler 2022 (arXiv:2202.07575); Pathak et al. 2022 FourCastNet (arXiv:2202.11214) — fine-tune LR ratios ·
Watt-Meyer et al. 2023 ACE (arXiv:2310.02074) — EMA 0.9999, +15% long-rollout ·
List, Chen, Bali & Thuerey 2025 (CMAME 433:117441; arXiv:2402.12971v2). Numbers from arXiv v2 (the CMAME body was not checked): in prediction setups, detached unrolling improves on one-step training by 41% on average and BPTT by a further 30%, and detached is better for small networks; in correction setups, detached 33% and BPTT 92% over one-step (v1, 2024-02-20, reported 23%/38% for correction setups, the figures this file quoted before 2026-10-01). Check every number against the open-access CMAME PDF before it enters the paper ·
Lippe et al. 2023 PDE-Refiner (NeurIPS, arXiv:2308.05732) — pushforward critique, spectral view ·
Sanchez-Gonzalez et al. 2020 GNS (ICML, arXiv:2002.09405); Stachenfeld et al. 2021 (arXiv:2112.15275) — noise injection (rejected locally) ·
Kelp et al. 2020 (JGR-A, doi:10.1029/2020JD032759) — chemistry: recurrent 24-step training (vs one-step) is load-bearing; one-step diverges exponentially ·
Kelp et al. 2022 (JAMES, doi:10.1029/2021MS002926) — coupled-state distribution dominates; species-without-sink drift ·
Bi et al. 2023 Pangu-Weather (Nature 619:533, arXiv:2211.02556); Bonavita 2024 (GRL, arXiv:2309.08473) — specialist models + boundary discontinuities ·
Liu, Kutz & Brunton 2022 HiTS (Phil Trans R Soc A 380:20210200) — per-Δt nets, U-curve, never ablated vs conditioning ·
Asensio Ramos et al. 2024 (MNRAS, arXiv:2406.02387) — single chemistry emulator over 7 time decades ·
Tancik et al. 2020 (NeurIPS, arXiv:2006.10739) — Fourier features: σ is the knob, fixed ≥ learned ·
Karras et al. 2024 EDM2 (CVPR, arXiv:2312.02696/networks) — normalize-then-unit-bandwidth pattern ·
Sturm & Wexler 2020/2022 (GMD 13/15); Sturm et al. 2024 (ACS EST Air, arXiv:2408.16109) — conservation: built-in fine, naive projection harmful ·
Bengio et al. 2015 (NeurIPS); Huszár 2015 (arXiv:1511.05101) — scheduled sampling improper (skipped) ·
Holdship et al. 2021 "Chemulator" (A&A 653:A76) — fixed-dt iterated chemistry emulator precedent.
