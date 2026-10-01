# Production config decisions — evidence and justification

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
  re-run from this repo.
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
configs, see the per-key table). If a processed dataset was
already built on the HPC after df2c44f, check the recorded value in the processed
dir's `preprocessing_summary.json` and rebuild if it was made with the 1e5 cap
(run.pbs skips preprocessing whenever the processed dir exists).

Stage-1 -> stage-2 checkpoint handoff (status note): stage 2 initializes from
`../models/stage1/checkpoints/last.ckpt`, i.e. the final-epoch EMA weights, not the
best-val_loss checkpoint. Whether the handoff should use last or best val_loss is an
OPEN decision to settle before launch; the config's checkpoint path stays unchanged
until it is. `testing/export.py` exports `checkpoints/best.ckpt`, matching the
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
real but specific to plain trunks at 7.6k params — a small-scale result that did not
transfer to the production-like architecture, exactly the failure mode the §"Evidence
base" policy anticipated. Because the feature was flag-gated, the correction is this
config flip; no code changes. The `FourierDtFeatures` code remains in src (validated,
export-safe) for the fallback below and the production-scale spot-check.

Consequences:
- Primary architecture = **the v3-lineage trunk unchanged** (10×2048 residual+LN+SiLU,
  predict_delta, plain scalar dt) — also the most consistent arm on the deployment
  metric and the only trunk validated at true production scale historically.
- New fallback (won the geometric metric both seeds, 1.3e-3/2.3e-3): plain-LN trunk
  WITH Fourier — set `model.mlp.residual=false` + `model.fourier_dt.enabled=true`.
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

## Evidence base and its limits

Decisions come from three sources: (a) the **local Robertson campaign**
(2026-06-09; 3-species stiff ODE sandbox, ~7.6k-param models, 2k–10k trajectories,
seeds 0–2, CPU), (b) **adversarially verified literature** (every load-bearing claim
checked against primary sources), and (c) **prior Auto-Chem/Chemulator runs**
(v2/v3 lineage, the published paper).

Two evaluation metrics were used throughout; they disagree in important ways:
- **geometric rollout** (consecutive log-grid steps, the sandbox's native task):
  seed noise ±3% — the *discriminator*.
- **fixed-dt rollout** (constant dt for 100 steps — what Canoe/VULCAN actually do):
  seed noise ±25% — the *catastrophe detector*. One-step/val metrics were NOT used
  for selection (Spearman ρ(one-shot, rollout) = 0.41).

**Transfer caveat (applies to every "local" row below):** production is ~100M params,
12 species conditioned on (P, T), ~62M samples; the sandbox is none of those. Per the
agreed policy, local wins were adopted only when replicated across seeds, mechanistically
sensible, AND cheap to disable; close calls ship both options. Sandbox models also bake
a per-step simplex projection into the rollout (production does not) — sandbox
*stability* numbers partially lean on it; relative comparisons should transfer better
than absolute ones. Scale checks: 18× params (7.6k→135k) and 5× data (2k→10k) showed
no inversion of any adopted finding.

## Confidence labels

- **high** — replicated locally + literature-aligned + scale-checked.
- **medium** — replicated locally (2–3 seeds) at sandbox scale; untested at production scale.
- **low / inherited** — carried from the v3-lineage templates or literature alone;
  not discriminated locally. Fallback or spot-check provided.

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
| **layout: ONE dt-conditioned model** (not dt bands) | single config pair over [0.1, 1e3] s | **Gate G2 (E2)**: band specialists collapsed at deployment-style fixed-dt *inside their own bands* (low band @ dt=0.01 s: 1.3–12.3 dex vs single model 0.33; high band @ dt=100 s: 1.8–2.0 dex vs 0.28–0.29; both seeds) while winning only marginally on their native grid task (2.09e-3 vs 2.41e-3, no handoff spike). Mechanism: dt-diverse training is augmentation against the (state, dt) distribution shift AR deployment creates. Literature: specialists' only proven benefit (Pangu-Weather HTA, HiTS) is composing large steps — unreachable at host-fixed dt; stitched specialists show documented boundary discontinuities (Bonavita 2024); single conditioned models proven over ≥7 time decades in chemistry (Asensio Ramos 2024) and lead times (Stormer). Consistent with the v3 narrow-dt-window deployment bottleneck. | medium-high (caveat: sandbox bands conflate time-segment with dt-band; production pathology likely milder → fallback shipped) |
| `model.fourier_dt.enabled` | **false (REVISED 2026-06-11)** | Original adoption (E1: 6.2× at 7.6k params, 3 seeds) was measured on PLAIN trunks. F1/F2 at ~130k params on the exact production-mirror residual trunk: Fourier+residual is the worst arm (fixed-dt mean 0.84/1.30 vs 0.45/0.42 without; geometric erratic 2.7e-2/4.8e-3; LR probe no rescue, 2 seeds). Either stabilizer alone works; together they interfere. See REVISION header. Spot-check recipe below remains the production-scale arbiter. | medium-high (replicated at the most production-like local scale) |
| `model.fourier_dt.num_freqs` | 16 | Tancik 2020: width secondary, saturates; 16–32 ample for a smooth scalar conditioner (diffusion models use 128 for far harder targets). Sandbox used 8 successfully; 16 adds margin at negligible cost. | medium (literature) |
| `model.fourier_dt.sigma` | 1.0 | EDM2 pattern: normalize the log-quantity to O(1) (Auto-Chem's dt_norm ∈ [0,1] already is), then σ≈1. Sandbox win used σ=3.0 on the same normalization — both in the working range; σ too high risks wiggly dt-interpolation at the deployment dt (Tancik). Spot-check: stage-1 val with σ ∈ {0.5, 1, 2} over the first ~20 epochs if desired. | medium (literature; exact value not locally discriminated) |
| `model.mlp.hidden_dims` 10×2048, residual, LN, SiLU, `predict_delta` | unchanged from v3 lineage | v3 (10×2048) reached val ~7e-4 and is deployed; LN value re-confirmed locally (MLP+LN 2.9× over MLP, E1); delta-head small-init is the sandbox/paper convention. Not re-tuned: out of local scope. | inherited |
| `preprocessing.dt_min/dt_max` | 0.1 / 1000.0 s | User deployment decision (Canoe dt≈6.4 s sits ~1.3 decades inside each edge — Fourier features interpolate well inside support, poorly outside; Tancik). | decision |
| `preprocessing.samples_per_source_trajectory` | 4 (was 1) | E2 mechanism: per-state dt diversity is the robustness lever; 4 independent (dt, t_start) draws per source trajectory maximize (state, dt) coverage from the same raws. | medium |
| `preprocessing.output_trajectories_per_file`, `pool_size` | 1e6 (consume all) | "Use the full dataset" (user decision); ~62M samples across ~160 raw files. | decision |
| `training.batch_size` | 2048 | Paper finding: >2048 degraded generalization; v2_bigger_batch (8192) plateaued at val ~9e-3. | high (production-scale evidence) |
| stage-1 `lr` / `wd` / scheduler / EMA 0.9995 / early-stop | template values | v3-lineage except the scheduler (v3 used `cosine_with_warmup`, see REVISION 2026-09-30; the shipped plateau schedule is not v3-lineage); converged to val ~7e-4 historically. EMA window rule (1/(1−d) ≈ 1–10% of updates, ACE uses 0.9999) — with full-dataset epochs, consider 0.9999; not locally discriminable. | inherited |
| **stage-2 model block = stage-1 model block** | 10×2048 (fourier_dt OFF) | Fixes the historical inconsistency: the now-removed `configs/stage2_dt_bands/` declared 8×1536 but strict-load 10×2048 checkpoints from wrong paths — would hard-fail at startup. | high (correctness fix) |
| `training.rollout_steps` (stage 2) | 10 | **E3 K_MAX sweep**: inverted-U on geometric — K=1: 6.5e-3–1.9e-2, K=5: 4.4e-3, K=10: 2.4e-3 (±3%, 3 seeds), K=20: 1.2–3.3e-2 + one 1.0-dex fixed-dt blow-up. Matches List et al. 2024 (detached unrolling counterproductive beyond ~6; BPTT helps further but K=10 is inside every published envelope — GraphCast 12 max). Training horizon ≪ deployment horizon is universal practice. | medium-high |
| `curriculum` 2→10, linear, 30 epochs | enabled | All multi-step SOTA use staged/ramped horizons (GraphCast +1/1k-updates 1→12; Stormer 1→4→8; NeuralGCM "critical"); sandbox used the equivalent ramp in every successful run. | high (literature + used throughout locally) |
| `autoregressive_training.skip_steps` | 1 | Brandstetter 2022 canonical pushforward (unroll 2, backprop last; "more stable"); composes with BPTT. Sandbox used skip 2 at K≥4 successfully; 1 is the literature value and wastes fewer target steps. | medium |
| `autoregressive_training.detach_between_steps` | **false (full BPTT)** | **E4**: BPTT ≈ detach locally (geometric 2.6–3.0e-3 vs 2.4e-3; fixed-dt 0.30–0.37 vs 0.37–0.55 — within noise, both seeds). Tie broken by literature: every SOTA AR forecaster backprops through the rollout (GraphCast, Keisler, FourCastNet, Stormer, NeuralGCM, ACE2); the only *chemistry* precedent (Kelp 2020, 24-step recurrent) is full BPTT; List 2024: BPTT 38% vs detached 23%. Memory at K=10/10×2048/batch-2048 is fine on GH200. | medium (local tie; high literature prior) |
| `autoregressive_training.loss_discount_gamma` | 1.0 (with BPTT) | **E3+E4 interaction**: under BPTT, γ=1.0 ≈ γ=0.9 (3.3e-3 vs 2.6e-3, within noise); uniform matches all SOTA practice. **Under DETACH, γ<1 is required**: uniform was 4–14× worse in both seeds (1.1e-2, 3.3e-2 vs 2.4e-3 at γ=0.9). Hence fallback recipe pins γ=0.9. | medium-high |
| `backward_per_step` | false | Only meaningful in the detached path; BPTT does one backward, so `backward_per_step=true` with `detach_between_steps=false` is rejected (since 2026-09-30). Fallback recipe sets true (memory). `detach=true` with `backward_per_step=false` gives the fallback's gradients but holds all K−skip step graphs until one backward (9 at K=10, skip 1, against 1 for the per-step fallback: BPTT's activation memory); rejecting that pair too is the author's call. | n/a |
| stage-2 `lr` | 3e-5 (= stage-1 peak / 10) | **E5**: staged pretrain→fine-tune at LR peak/10 gave the best fixed-dt result of the whole campaign (0.256 vs 0.379 from-scratch); LR peak/3 (1e-4) doubled geometric error with no fixed-dt gain. Literature: AR fine-tune LR is 5–3000× below pretrain peak (FourCastNet /5, Keisler /10 per stage, Stormer /100–/1000, GraphCast /3000); /10 is at the aggressive-but-precedented end, and the old template's 2e-5 is also fine. | medium-high |
| stage-2 `wd` | 1e-5 (= stage 1; old template's 0.005 dropped) | No SOTA fine-tune cranks WD ×500 at the phase boundary; the 0.005 value was untested (stage 2 never ran). Keep optimizer geometry constant except LR. | medium (literature/consistency) |
| stage-2 `lambda_z_mse` 0.25, EMA 0.999, early-stop 20, plateau warmup 5 | template values | Inherited from the stage-2 template; not locally discriminated. EMA re-warms automatically at stage start (fresh callback state); ACE evidence says EMA helps long-rollout metrics up to 15%. | inherited |
| `runtime.torch_compile.compile_open_loop_unroll` | stage-1 true / stage-2 false | The flag affects only the validation open-loop rollout (`trainer.py` `_eval_step`), never the AR training loop, and eval K ignores the curriculum. Stage 2 disables whole-unroll compilation because a fullgraph K=10 unroll of the 10x2048 trunk is a 10x-deep inductor graph (long compile time; recompiles on a ragged final validation batch with dynamic=false); eval still runs the compiled forward step per step. Stage-1 true is redundant-but-harmless at eval K=1. | n/a (throughput only; no effect on math) |
| `runtime.torch_compile.mode` | stage-1 `reduce-overhead` (REVISED 2026-09-30, was `default`) / stage-2 `default` | Stage 1 replays its compiled training step and validation rollouts as CUDA graphs, which cuts host-side launch work: on a CPU-bound proxy (10x256 trunk, logging fixes applied) the stage-1 CPU time per step fell from 3.85 to 2.82 ms (audit efficiency study). On daw's RTX 6000 Ada stage 1 is GPU-bound and wall time did not change (-0.9%, inside noise); a gain is expected only where stage 1 is CPU-bound, e.g. on the GH200 (not measured). Numerics are identical: bitwise 30-step fingerprints (live/EMA weights, every loss, val_loss), and a 3-epoch run that replays every training and validation graph (ragged last batch included, after the EMA weight swap) matches `default` bit for bit. Compile variants with a ragged last validation batch: forward step 1, validation rollout 2 (the recompile limit is 8). Under CUDA graphs the next step's replay reuses the previous step's `.grad` memory, so nothing may read gradients across steps; nothing does (`runtime.log_grad_norm` reads the gradients in both stages, but only within the step, before the optimizer step; since 2026-09-30). Stage 2 keeps `default`: it is GPU-bound (no expected gain), and its no-grad pushforward output feeds the grad-mode step, which the study expects to conflict with CUDA-graph output reuse (inferred, not run). | n/a (throughput only; bitwise-identical) |
| `preprocessing.num_workers` | 16 (REVISED 2026-09-30, was 64) | Preprocessing runs on daw, where one job uses at most 16 cores; the fan-out is also capped at the number of raw files and of cores the process may run on. Outputs do not depend on the worker count (per-file RNG and shard names): 1 and 4 workers gave identical arrays and normalization.json on the synthetic set (audit T1 check). Supersedes the 2026-07-16 "num_workers 64 for parity" note. | n/a (throughput only) |
| `training.early_stopping.min_delta` with ES `patience`, plateau `patience` / `threshold` | min_delta 0.0 (REVISED 2026-09-30, was 1e-5); ES patience 30 / 20; plateau patience 15 / 10, threshold 1e-3 / 1e-4 relative (`threshold_mode` "rel", a named constant in trainer.py) | ES and the plateau observe the same epochs: both ignore the LR warmup and the curriculum ramp and start counting at the first epoch after both (warmup-era epochs stay eligible for best.ckpt). With min_delta 0 every plateau improvement is also a new ES best, and plateau patience + 1 < ES patience (16 < 30, 11 < 20), so once counting has started ES cannot stop before an LR halving. Before, stage 1 could stop at peak LR with zero LR cuts (audit harness, constructed curves with the best inside the warmup: stopped at epochs 35 and 54; with T1 neither stops). Under steady slow improvement `max_epochs` (400 / 200) becomes the binding budget and the relative thresholds decide whether slow progress cuts the LR; tuning them is the author's. Alternative pending the author: both monitors keep the warmup-era best (an annealing-policy choice). A pre-fix run cannot be resumed with these values (the resume config check refuses changed keys). | inherited (min_delta and the shared window: audit) |
| `runtime.accelerator` / `runtime.devices` | "cuda" / 1 (REVISED 2026-09-30, were "auto") | Explicit single device: "auto" falls back to CPU silently and Lightning's "gpu" alias tries MPS first; a CUDA request with no visible GPU now fails at startup. The launcher picks the GPU with CUDA_VISIBLE_DEVICES (devices=1 means visible index 0). Multi-GPU/DDP is unsupported. | n/a |
| `system.device` / `training.pin_memory` | "cpu" / true | Every split is preloaded into host RAM (a CUDA preload would put the whole dataset in GPU memory) and batches are pinned for the copy; pin_memory=true with a CUDA preload is rejected. Window indices come from a CPU generator, so the window sequence does not depend on the preload device. Preload budget: a host preload (train + validation, sized from the shard headers) larger than 0.5 × MemAvailable fails at startup instead of being OOM-killed while filling (`_PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE` in src/main.py, not a config key). MemAvailable is a snapshot on a shared host, not a guarantee. The 0.5 is pending the author: at ~785 GB MemAvailable (daw, T1 smoke log) it caps the preload at ~393 GB, below the ~481 GB the audit infers for all 160 raw files, so a full-data run would be refused today. | decision pending (the fraction) |
| `runtime.gradient_clip_val` / `runtime.log_grad_norm` | 1.0 / false | Clip 1.0 is inherited; under AdamW, clipping reweights batches, it does not lower the LR. null is the only "no clipping" value (0 and negatives are rejected). `log_grad_norm` (a required key since 2026-09-30) logs per epoch the mean pre-clip total gradient norm (`grad_norm`) and the fraction of steps where clipping fired (`grad_clip_frac`), in both stages, summed on the GPU. | inherited |
| `precision.*` | float32, amp_mode "true" | The preload is always float32 (`precision.preload_dtype` removed 2026-09-30: bf16 storage moved states by a median 3.1e-3 dex in the audit, above the ~7e-4 target). loss_dtype must be float32 or float64. Under amp_mode "mixed", model_dtype and input_dtype must be float32 and `model.predict_delta` true: a bf16 input, or a whole next state coming out of the bf16 trunk, would carry the rollout state at bf16 resolution. Under autocast the input block and the delta head run in fp32 (fp32 and bf16-true runs are unchanged); bf16-mixed with fused AdamW and clipping trains. bf16-true is kept but lossy: the rollout state is carried in bf16. | decision (float32 default) |
| `preprocessing.drop_below` | 1e-30 (= normalization.epsilon) | `drop_below >= epsilon` is required (2026-09-30): every kept raw value is then >= epsilon, so the epsilon floors in the log transforms never change a kept value. | n/a (contract) |
| **rejected: input-noise injection** | not shipped | **E6**: 4/4 noise runs worse, 3/4 catastrophic at fixed-dt (12.4–14.1 dex; σ ∈ {0.01, 0.03} z-space, with and without pushforward, 2 seeds) — *against* the GNS/Stachenfeld prior. No trainer option added. If ever revisited, sweep σ ≤ 1e-3 and re-test. | high (locally decisive) |
| **rejected: K=20 detached training** | — | E3: consistently worse + one 1-dex blow-up; List 2024 agrees. (K>10 with BPTT untested locally — plausible future direction, not a default.) | medium |
| **not adopted: inference-time renormalization** | off | Sturm 2024: naive projection degrades radicals/trace species (exactly what dominates a log-standardized loss); weighted projection neutral-to-positive. Track conservation drift as an EVAL metric; if the host solver needs hard conservation, the stoichiometric-output-layer route (Sturm & Wexler 2022) is the accuracy-neutral fix. | high (literature) |

## Documented alternatives (evidence-backed; not shipped as default configs)

1. **Banded layout** — two-band stage1+stage2 over {1e-1–1e1, 1e1–1e3} s. Rejected by
   Gate G2 (band specialists collapsed at deployment-style fixed dt) and **removed** from
   the repo on 2026-06-15 (the former `configs/production/fallback_banded/`). Recover from
   git history only if the single model shows a measured deployment-dt gap that a
   deployment-dt-concentrated stage-2 (below) cannot close.
2. **Detached stage-2 recipe** (lower memory, equally validated locally): in
   `configs/stage2.json` set `detach_between_steps=true`, `backward_per_step=true`,
   `loss_discount_gamma=0.9`. Do NOT run detached with γ=1.0 (E3: 4–14× worse).
3. **Deployment-specialized fine-tune** (if production rollout at dt≈6.4 s underperforms):
   rather than banding, run a second stage-2 pass with preprocessing dt_min/dt_max
   narrowed to ~[2, 20] s from the *same* stage-1 checkpoint (FuXi-style). Keeps one
   model family, no band-boundary discontinuity.
4. **Autoencoder trunk** — removed from src on 2026-09-30; recover from git history (tag
   `pre-audit-2026-09-29`). Rationale kept: Fourier-Flow-map was worse + 5.5× seed spread at
   sandbox scale (E1) but small-scale bottleneck behavior may not transfer; documented, not pursued.

## Cheap production-scale spot-checks (optional, per the transfer-risk policy)

Each ~20 stage-1 epochs (~5% of a full run) on one GH200, comparing val curves:
- **fourier_dt on/off** — the single most load-bearing adoption.
- **fourier_dt.sigma ∈ {0.5, 1, 2}** — only if the on/off check shows sensitivity.
- **stage-2, first 10 epochs**: BPTT+γ=1.0 vs detached+γ=0.9 — verifies the E4 tie at scale
  before committing the remaining 190 epochs.

## Verification performed (2026-06-09)

- **End-to-end smoke** (`testing/run_local_smoke.sh`): synthetic raw → preprocessing → stage-1
  (2 epochs) → stage-2 default (BPTT) AND stage-2 fallback (detached+γ=0.9), with the
  weights_only EMA handoff. PASSED on this machine.
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
  legacy configs unaffected.

## HPC validation protocol (acceptance gates for the trained models)

1. Stage-1: val one-jump loss ≤ v3 reference (~7e-4).
2. Stage-2 (the deliverable's real test): held-out test-split **open-loop rollouts at
   fixed dt = 6.4 s for K ≥ 100**, plus dt = 0.5 s and dt = 100 s; require 100% finite
   trajectories, error growth that flattens after the transient (the sandbox shape:
   front-loaded, then plateau — no super-linear growth past ~step 20), and 90th-pct
   per-species log10 error well below the paper's one-shot-trained 56.6% @ 99 steps
   (target < 10%). Track species-wise bias drift, especially any species lacking strong
   chemical sinks (Kelp 2022's HNO3 lesson), and element-conservation drift as metrics.
3. Cross-check vs mini-chem in the Canoe harness (`Canoe-Chem`, dt≈6.4 s) before swap-in.

## Key citations (all verified against primary sources 2026-06-09)

Brandstetter, Worrall & Welling 2022 (ICLR, arXiv:2202.03376) — pushforward trick ·
Lam et al. 2023 GraphCast (Science 382:1416, arXiv:2212.12794) — BPTT, 1→12 curriculum, LR 3e-7 ·
Nguyen et al. 2024 Stormer (NeurIPS, arXiv:2312.03876) — single dt-conditioned model, staged K, LR /100–/1000 ·
Keisler 2022 (arXiv:2202.07575); Pathak et al. 2022 FourCastNet (arXiv:2202.11214) — fine-tune LR ratios ·
Watt-Meyer et al. 2023 ACE (arXiv:2310.02074) — EMA 0.9999, +15% long-rollout ·
List, Chen, Bali & Thuerey 2024/25 (CMAME, arXiv:2402.12971) — BPTT 38% vs detached 23% ·
Lippe et al. 2023 PDE-Refiner (NeurIPS, arXiv:2308.05732) — pushforward critique, spectral view ·
Sanchez-Gonzalez et al. 2020 GNS (ICML, arXiv:2002.09405); Stachenfeld et al. 2021 (arXiv:2112.15275) — noise injection (rejected locally) ·
Kelp et al. 2020 (JGR-A, doi:10.1029/2020JD032759) — chemistry: recurrent BPTT is load-bearing; one-step diverges exponentially ·
Kelp et al. 2022 (JAMES, doi:10.1029/2021MS002926) — coupled-state distribution dominates; species-without-sink drift ·
Bi et al. 2023 Pangu-Weather (Nature 619:533, arXiv:2211.02556); Bonavita 2024 (GRL, arXiv:2309.08473) — specialist models + boundary discontinuities ·
Liu, Kutz & Brunton 2022 HiTS (Phil Trans R Soc A 380:20210200) — per-Δt nets, U-curve, never ablated vs conditioning ·
Asensio Ramos et al. 2024 (MNRAS, arXiv:2406.02387) — single chemistry emulator over 7 time decades ·
Tancik et al. 2020 (NeurIPS, arXiv:2006.10739) — Fourier features: σ is the knob, fixed ≥ learned ·
Karras et al. 2024 EDM2 (CVPR, arXiv:2312.02696/networks) — normalize-then-unit-bandwidth pattern ·
Sturm & Wexler 2020/2022 (GMD 13/15); Sturm et al. 2024 (ACS EST Air, arXiv:2408.16109) — conservation: built-in fine, naive projection harmful ·
Bengio et al. 2015 (NeurIPS); Huszár 2015 (arXiv:1511.05101) — scheduled sampling improper (skipped) ·
Holdship et al. 2021 "Chemulator" (A&A 653:A76) — fixed-dt iterated chemistry emulator precedent.
