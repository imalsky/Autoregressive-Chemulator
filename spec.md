# Autoregressive Chemical Kinetics Emulator Specification

Version: 1.10
Date: 2026-10-01
Audience: Professional collaborators and maintainers

## 0. Execution Environment
1. The canonical machine is daw (shared SETI Institute workstation, 2x RTX 6000 Ada); the PBS/NAS batch scripts are retired.
2. The canonical environment is the uv venv `.venv` in the repo root: Python 3.12.14 (`.python-version`) with the exact pins of `requirements.lock` (9.1). `python` in commands and run lines means `.venv/bin/python` (or `.venv` activated).
3. Training, preprocessing and the post-training steps run through `run_daw.sh` (one GPU via `CUDA_VISIBLE_DEVICES`, refusing a second GPU for the same user; the process pinned to 16 CPUs on that GPU's NUMA node by default; OMP/MKL/OPENBLAS/NUMEXPR threads set to that count; `TMPDIR` and the inductor/triton caches in the user's space). `bash run_daw.sh smoke` runs the smoke test (4.4.4) in the same environment.
4. Running outside `.venv`, or training outside `run_daw.sh`, is non-canonical.

## 1. Product Goal
1. Build and maintain an accurate, fast autoregressive chemical kinetics emulator.
2. Prioritize inference quality and runtime speed.
3. Use `val_loss` as the primary model-selection metric.

## 2. Project Scope
1. Scope includes the entire project lifecycle.
2. Lifecycle includes preprocessing, training, validation, export, and inference tooling.
3. Data/asset directories are not source-edit targets: `data/`, `figures/`, `misc/`, `models/`.
4. Runtime artifacts may still be generated under artifact directories (for example `models/` and `figures/`).

## 3. Engineering Principles
1. Fail fast in canonical preprocessing, training, and export contracts.
2. Diagnostic scripts under `testing/` may be more permissive when they are not release gates.
3. No silent behavior changes.
4. Hard-fail on malformed config, schema mismatch, ambiguous keys, and unsupported modes in canonical runtime paths.
5. Keep logic explicit and simple.
6. Avoid magic numbers in logic bodies.
7. Place stable defaults as named module-level constants in each executable script.
8. There is no default config file; all entrypoints require `AUTOCHEM_CONFIG_PATH` and hard-fail if it is unset. The testing tools select their run directory (the config's `paths.work_dir`) through the same variable and one resolver (`testing/run_dir.py`).
9. Required config keys are read directly (absence is an error); only optional feature blocks may carry named-constant defaults.

## 4. Canonical Workflow

## 4.1 Preprocessing
1. Preprocessing is canonical via `processing/preprocessing.py`, selected by `AUTOCHEM_CONFIG_PATH` (required).
2. Fixed `dt` mode is removed.
3. Variable `dt` is mandatory.
4. Required preprocessing `dt` config uses `dt_min` and `dt_max` only.
5. `dt_sampling` is fixed to `loguniform`.
6. Log-time/log-value interpolation is mandatory.
7. `drop_below` rejection is mandatory.
8. Raw schema is considered stable and non-evolving.
9. Species datasets must be scalar-per-time and only use shapes `[T]` or `[T, 1]`.
10. Multi-column or higher-rank species datasets are unsupported and must hard-fail.

## 4.2 Training
1. Training entrypoint is `src/main.py`, selected by `AUTOCHEM_CONFIG_PATH` (required; there is no default config file).
2. Training mode is the only supported runtime mode.
3. Post-train automatic test execution is removed.
4. Validation remains enabled during training.
5. Training is a two-run workflow:
6. Stage 1 run uses one-jump training with `rollout_steps = 1`.
7. Stage 2 run uses autoregressive multi-step training with `rollout_steps = N`, where `N > 1`.
8. Stage 2 `N` is user-configured per run and not fixed by the codebase.
9. The order of the two runs is a canonical workflow guideline, but the stage-2 handoff is guarded: a `weights_only` load requires the source run's `COMPLETED` marker and a matching normalization fingerprint (`norm_sha256`), and caps the run's peak LR at min(`training.optimizer.lr`, the source checkpoint's optimizer LR); a checkpoint without that optimizer state is refused.
10. Resume/checkpoint options are retained.
11. Fresh runs still require an empty work directory unless explicitly resuming; a work directory whose only entry is `train.log` (left by a failed start) counts as empty. The training process holds an exclusive flock on the work directory itself (no lock file is created), so a second training process on the same work directory fails at startup.
12. Curriculum support is retained.
13. Autoregressive controls (`skip_steps`, detach behavior, backward mode) are retained.
14. Optimizer support is restricted to `AdamW` only.
15. Scheduler support is restricted to:
16. `ReduceLROnPlateau` with a linear warmup (`training.scheduler.*`), the only scheduler; there is no scheduler-type key.
17. `cosine_with_warmup` is unsupported (removed 2026-09-30).
18. `torch.compile` support is retained and optional.
19. Compile default remains enabled: stage 1 uses `runtime.torch_compile.mode = "reduce-overhead"` (CUDA graphs), stage 2 `"default"` (evidence: `docs/DECISIONS.md`).
20. Relative `runtime.checkpoint` paths are resolved against the config-file directory.
21. `runtime.checkpoint` must exist and be a file; otherwise training must hard-fail.
22. Training framework imports must use `lightning.pytorch`.
23. Direct source imports from `pytorch_lightning` are unsupported and must not appear in the repository.
24. Trainer/logger integration targets the current `lightning.pytorch` API directly; legacy PyTorch Lightning compatibility shims are not part of the supported codebase.
25. The production configs are `configs/stage1.json` (one-jump pretrain) and `configs/stage2.json` (autoregressive fine-tune); these are the only files in `configs/`.
26. The production layout is a single dt-conditioned model. The band-specialist (per-`dt`-range) layout was evaluated and rejected; see `docs/DECISIONS.md`.
27. `training.ema` and `training.early_stopping` are required config blocks, each gated by its own `enabled` flag.
28. `model.fourier_dt` and `training.autoregressive_training.loss_discount_gamma` are optional, validated feature blocks; absent keys use named-constant defaults that reproduce legacy behavior.
29. `precision.amp_mode` is Lightning's precision suffix: `"true"` means `*-true` precision with no autocast (required for float32 and float64 compute; bfloat16 with `"true"` is pure bf16, a lossy mode that carries the rollout state in bf16); `"mixed"` enables autocast. TF32 is the separate `precision.allow_tf32` key, forced off by `runtime.deterministic=true`.

## 4.3 Export and Inference
1. Export contract is physical-space one-step inference.
2. Export signature is:
3. `y_next_phys = model(y_phys, dt_seconds, g_phys)`
4. Tensor shapes are:
5. `y_phys`: `[B, S]`
6. `dt_seconds`: `[B]`
7. `g_phys`: `[B, G]`
8. `y_next_phys`: `[B, S]`
9. Dynamic batch `B` is mandatory.
10. Variable per-step `dt` support is mandatory.
11. Baked normalization in export is mandatory.
12. Export naming standard is mandatory:
13. `export_{device}_dynB_1step_phys.pt2`
14. Canonical requested targets are CPU and CUDA.
15. `testing/export.py` exports every requested target; a requested target that is not available is an error (request `cpu` only to export CPU only).
16. MPS is unsupported; export targets are `cpu` and `cuda` only.
17. Export metadata source of truth is embedded `metadata.json` inside the `.pt2` artifact.
18. No sidecar metadata JSON is required.
19. For `torch.export` artifacts, inference loaders must not call `model.eval()` or `model.train()` on `ep.module()` because these mode-switch APIs are unsupported; run inference under `torch.inference_mode()` instead.
20. The run is selected by `AUTOCHEM_CONFIG_PATH` (its `paths.work_dir`); export runs only for a finished run (`COMPLETED`) whose checkpoint normalization fingerprint (`norm_sha256`) matches its processed data.
21. Units (metadata `units`): `dt_seconds` in s; `g_phys` P in dyn/cm^2 (barye) and T in K; `y_phys` and `y_next_phys` are species mixing ratios in the training-data convention (n_i / n_total over all gas of the generator run; the training initial gas held only the tracked species, so they sum to 1 at t=0 and at most 1 later; train-split distribution of the sum: metadata `species_sum`). No unit conversion is applied anywhere.
22. Species inputs are not range-checked: values at or below `epsilon` are floored at `epsilon` before log10. Outputs are neither clipped nor renormalized and use the same denominator as the inputs.
23. Out-of-range rows (metadata `oor_policy: "nan_row"`): a row whose `dt_seconds`, P or T lies outside the train-split box in metadata `oor_box` by more than `oor_tol` of the range (in the normalizer's coordinate: log10 for dt and log-* methods, linear otherwise), or whose inputs are not finite, returns NaN in every species. A NaN row means only that; in-range rows are computed exactly as without the rule.
24. The AOTI package (`testing/aoti_export.py`) does no input validation: callers pass CPU float32 C-contiguous `[B, S]`, `[B]`, `[B, G]` tensors with one `B` inside metadata `dynamic_batch` (metadata `host_input_contract`). It is native code for the build CPU (metadata `build_target`) and is installed only when it matches the fresh export.

## 4.4 Diagnostic and Testing Scripts
1. `testing/` holds diagnostic and deployment utilities; they are not release gates, except the integration gate `testing/run_local_smoke.sh` (4.4.4).
2. `testing/export.py` and `testing/aoti_export.py` produce/validate deployment artifacts and follow the strict no-fallback contract.
3. The plotting/inspection diagnostics (`testing/predictions.py`, `testing/bench_speedup.py`, `testing/training_logs.py`, `testing/efficiency_summary.py`, `testing/profile_training.py`) may use explicit best-effort behavior for portability and local usability.
4. `testing/run_local_smoke.sh` (run as `bash run_daw.sh smoke`) is the integration gate: the production configs, shrunk, on synthetic data through preprocessing (its two reject paths), both stages with the production compile settings, a deliberate extension and a refused re-run of a finished run, the `best.ckpt` handoff, both stage-2 recipes, the Fourier-dt fallback pair and the gated export with a reload check. It asserts the outcomes (finite losses, epoch counts, `COMPLETED`, `last.ckpt` at the final epoch, a ragged last validation batch, compiled-variant counts, weights that change, NaN rows out of range) and fails on the first wrong one. Run it before committing changes to `src/`, `processing/` or the export.
5. `testing/eval_rollout_error.py` (rollout error vs step, against persistence) and the planned fixed-dt evaluation-set generator are diagnostics under 6.3: the stage-2 rollout target in `docs/DECISIONS.md` is a reported target (per-species p90 in dex at a stated step and dt; its threshold is an open author decision), not a hard gate, and the provisional stage-1 gate (skill against persistence) is printed as pass/fail lines that do not change the exit status. The script reads the build's raw files to recover window-start times. If the author makes either a gate, both move under the strict contract of 4.4.2: a non-zero exit on NaN percentiles, a K below the requested K, non-finite trajectories, truncated rows or a missed target.

## 5. Data and Schema Contracts
1. Raw input data schema is stable and assumed unchanged.
2. Dataset path/key ambiguity is a hard error.
3. Globals are fixed per trajectory and scalar-only.
4. Global variable set is fixed and mandatory:
5. `["P", "T"]` in this exact order.
6. `G = 2` is mandatory for production workflows.
7. Species/global ordering must match exactly between config and `normalization.json`.
8. Processed shard schema remains fixed:
9. `y_mat` float32
10. `globals` float32
11. `dt_norm_mat` float32
12. `normalization.json` is generated by preprocessing and consumed downstream.
13. Split strategy should stay simple; leakage minimization beyond current behavior is not a requirement.
14. Global normalization methods are restricted to `standard`, `min-max`, `log-standard`, and `log-min-max`.
15. Global normalization method `identity` is unsupported and must hard-fail.

## 6. Metrics and Model Selection
1. Keep current loss and metric definitions.
2. Primary checkpoint/selection metric is `val_loss`.
3. No additional hard acceptance thresholds by species/horizon are required.
4. No physical-constraint enforcement metrics are required.
5. Benchmark and VULCAN scripts are diagnostic, not release gates.
6. Validation scores a fixed window set defined by the processed data and `dataset.val_samples` only (an even subset of the enumerated (trajectory, start) windows, no random draws), so it does not change across epochs, resumes, seeds, batch sizes or preload devices.
7. `val_loss` is comparable across seeds and batch sizes on the same processed data at equal `rollout_steps` and `skip_steps`; it is not comparable across processed datasets or across those recipe settings.

## 7. Logging and Outputs
1. Maintain high-quality operational logging.
2. Persist resolved config for each run.
3. Persist training metrics (`metrics.csv`) for each run.
4. Persist checkpoints according to configured checkpoint policy.

## 8. Non-Goals
1. Enforcing physical constraints in-model or post-hoc.
2. Supporting fixed-`dt` preprocessing mode.
3. Supporting multiple runtime modes beyond training.
4. Requiring unit tests, or integration tests beyond the smoke gate (4.4.4).
5. Treating VULCAN comparison or benchmark scripts as mandatory acceptance gates.

## 9. Developer Tooling Notes
### 9.1 Canonical Environment Installs
1. From the repo root: `uv venv --python 3.12.14 && uv pip sync requirements.lock`.
2. `requirements.lock` is the reproducibility source (the daw venv freeze; torch 2.8.0+cu126 by wheel URL) and includes the developer tools (ruff 0.16.9, pyflakes, vulture); `pyproject.toml` carries only minor-series bounds.
3. The codebase targets the `lightning.pytorch` namespace; direct imports from `pytorch_lightning` are non-canonical even if that package is present transitively in the environment.

### 9.2 Dead Code Search Workflow
1. Run checks from the project root with the venv's tools.
2. Canonical dead/unused checks:
3. `.venv/bin/pyflakes $(git ls-files '*.py')`
4. `.venv/bin/ruff check $(git ls-files '*.py')` (rule set pinned in `pyproject.toml` `[tool.ruff.lint]`: E4, E7, E9, F)
5. `.venv/bin/vulture $(git ls-files '*.py')`
6. Optional high-confidence-only Vulture pass:
7. `.venv/bin/vulture --min-confidence 100 $(git ls-files '*.py')`

### 9.3 Editing Rules For Dead Code Fixes
1. Remove truly unused locals/imports/functions rather than suppressing diagnostics.
2. If protocol signatures require unused parameters (for example context manager or framework hook signatures), rename them with leading underscores only when the caller passes them positionally. Keep the names a framework passes by keyword (Lightning calls `setup` and `teardown` with `stage=`, so `EMACallback.setup`'s `stage` keeps its name), optionally with a targeted `# noqa` comment; a dead-code tool's report on such a parameter is a false positive.
3. Treat framework callback methods and model `forward` methods as potential false positives in static dead-code tools; verify before deleting.
4. Keep parser argument removal policy: no `argparse`; use config values or explicit module-level constants.
