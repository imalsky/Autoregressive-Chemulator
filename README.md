# Auto-Chem (Autoregressive Chemulator)

A neural surrogate for the gas-phase chemical kinetics of exoplanet atmospheres, built to be called once per cell and
time step inside a general circulation model (GCM). Given a cell's composition at pressure P and temperature T, one
call advances it by a time step dt. It is trained on 0-D VULCAN chemistry runs (a C-H-O-N network of 53 species, helium
included) and is applied autoregressively: its output is the next call's input.

The code for the published 12-species model is tag
[`paper-v1`](https://github.com/imalsky/Autoregressive-Chemulator/tree/paper-v1). This branch holds the current model,
which tracks a closed state and conserves the element ratios.

## The model

**State.** 25 entries per cell, kept as normalized log10 mixing ratios z = (log10 y - mu) / sd (per-entry mu, sd from the
normalization file; y floored at 1e-30):

- the 12 reported species C2H2, CH4, CO2, CO, H2O, H2, HCN, H, N2, NH3, OH, O;
- He and 7 more tracked species: C, N, CH3, H2CN, C2H6, CN, O2;
- 5 reservoirs over the other 33 network species: C_res, N_res, O_res and H_res (the atoms of each element held in
  them) and M_res (their molecule count).

**One call** (`lib/models.py`, head `relaxclosedr`):

1. A residual MLP trunk (6 layers of 2048, SiLU, LayerNorm) maps z and the normalized globals (log10 P, T) to a target
   offset r and a rate s per entry. It never sees dt.
2. Relaxation, in float64: z~ = z + (1 - exp(-k dt)) (q - z), with q = clamp(z + r) inside the physical box
   1e-30 <= y <= 1 and log10 k = c + h tanh(s / h) in [-12, 3]. dt enters only through the exponential.
3. Closure: y~ is moved onto the input's element ratios to hydrogen (C/H, N/H, O/H, He/H) by y' = y~ exp(A lambda), with
   lambda from a damped Newton solve in float64 (`lib/elemproj.py`), and normalized so the molecules sum to 1. A row
   whose solve does not converge is re-solved by Levenberg-Marquardt (bench option `--closed-fallback lm`) or keeps its
   input.

The network runs in float32; the state and the projection are float64, and the caller adds the returned increment to
its float64 state. The current model, CBg, has 21.2 M parameters and costs about 3.2 us per cell per call at 10^5 cells
on one RTX 6000 Ada GPU (CUDA graph, 32 Newton iterations; about 5.3 us with the LM rescue on every row), measured with
`bench/speed.py`.

## Training

`lib/train.py` trains in two stages from a GPU-resident cache of VULCAN trajectories (each trajectory stored on one
log-time grid of 100 rows; labels between rows are log-log interpolated and projected back onto the trajectory's element
ratios):

- **Stage 1**: one-step windows, dt log-uniform in [dt_min_s, dt_max_s].
- **Stage 2**: four terms whose model calls run in lockstep: the one-step windows; long-jump pairs (z(t), z(t + D)) with
  D log-uniform in 0.1 s to 1e7 s; grid chains of 20 stored log-time intervals from t >= 1e3 s, each made of 1-4 model
  calls, trained through time; changing-condition chains with GCM-like mixing between calls (0-80 calls without
  gradient, then 8-16 with). Gradients are clipped per row inside the rollouts and globally (norm in float64); steps with
  a non-finite gradient are skipped.

Loss: log10 MAE plus a z-space MSE. AdamW, linear warmup then cosine decay; the EMA weights are saved. Stage 2 starts from
stage 1's EMA weights.

    source run/env.sh
    ARCH_CACHE=$CHEM_SCRATCH/arch_gcmU4_res HARNESS_NORM=$CHEM_SCRATCH/arch_gcm17m_res/norm.json \
        $PY lib/train.py <run dir>          # <run dir>/config.json: keys of train.DEFAULTS (unknown keys are refused)

A run writes `stage1.pt` / `stage2.pt`, `log.csv`, `done.json` and `manifest.json` (checkpoint hashes, the resolved config,
a copy of the normalization file and the code version; the bench refuses a normalization file that differs).

**CBg's recipe** (the two run folders' `config.json`):

- stage 1 (run `e13s_CBS1_s2`): `hidden` 6 x 2048, `batch` 8192, `s1_steps` 1,000,000, `s1_lr` 6e-4, `s1_warmup` 20,000,
  `dt_min_s` 10, `dt_max_s` 500, `seed` 2, `traj_weights_file` w_U4_s1.npy, `x0_file` run_gcm_m18 (`x0_anchor_p` 0.5,
  `x0_jump_p` 0.25);
- stage 2 (run `e13s_CBg_s2`): `init_from` the stage-1 file, `s2_steps` 200,000, `s2_warmup` 4000, `jump_w` 0.2
  (`jump_batch` 2048), `grid_w` 1.0 (`grid_batch` 1024, `grid_end_frac` 0.5), `chain_w` 0.5 (`chain_batch` 1024, K 8 -> 16
  and pre-roll 0 -> 80 over 15,000 steps, two chain sets), `grid_row_clip` and `chain_row_clip` 0.01, `clip_norm_fp64`,
  `skip_nonfinite`, `fused_proj`, `traj_weights_file` w_U4_s2.npy, `x0_file` run_gcm_m18 and run_gcm_z22k.

Stage 1 took about 12 h and stage 2 about 21 h on one RTX 6000 Ada. Two training runs of one config are not bitwise
identical (CUDA multinomial draws and PyTorch's run-time kernel tuning), so a retrain gives an equivalent model, not the
same weights. CBg's stage 2 continued from step 10,000 of a run with the same settings (recorded in its manifest).

## Benchmarks and scoring

- `lib/lhbench.py`: rollouts on the GPU. Arm H starts from a VULCAN tag's own states at fixed T, P and dt; arm P follows
  changing-condition paths (T, P and dt change every call, with recorded mixing). The decision benches use
  `--modes P3 --closed-iters 32 --closed-fallback lm --lm-adaptive` (and `--startup-substeps 5:1` on arm H). Results
  differ by up to 7e-4 dex across batch shapes, so a bench keeps its rows per call.
- `run/bench_dev.sh`: the development sets (fixed-condition sets as one combined arm-H bench, five path sets, the T2
  snap-back test, small steps from evolved states, the mini-chem cases); `run/bench_candidate.sh` adds speed, a
  1000-day run (`bench/cyc1000.py`) and the judge.
- Step-size sweep: `run/run_sweep.sh` (`bench/sweep/sweep.py`: 606 jobs x 9 step sizes from 1 to 500 s x 2 start ages,
  10,000 calls) and `bench/sweep/score_sweep.py`.
- `bench/score_minichem.py`: the six mini-chem paper cases (Tsai et al. 2022, Fig. 4) against VULCAN, for model benches and
  for stored mini-chem predictions.
- Scoring (`lib/scorerule.py`): per-gas floors (1e-10 for H2O, CO, CO2, CH4, NH3, HCN, C2H2; 1e-8 for H, OH, O; 1e-6 for
  H2, N2); a point counts where the truth or the prediction is above its floor and resolved (>= 1e-2 cm^-3); the error is
  |log10 max(pred, floor) - log10 max(truth, floor)|. Boards and rules: `bench/scoreboard.py`, `bench/path_board.py`,
  `bench/board6.py`, `bench/rules_t5.py`, `bench/judge.py`, `bench/pass_v2.py`.
- `tests/bitcheck_train.py` (training, bit for bit against a reference code copy) and `tests/score_regress.py` (scorers
  against stored results) are regression checks; they need the stored runs and benches.

## Layout

    lib/        models.py (the head), train.py, data.py (cache and samplers), chains.py, elemproj.py, lhbench.py,
                scorerule.py, rawtraj.py, paths.py (every data location)
    bench/      scorers, boards, decision rules, speed, the 1000-day run, the step-size sweep (bench/sweep/)
    datagen/    cache builders (generator packs -> 12-, 53-species and 25-entry caches), draw weights, T2 kick tags
    run/        env.sh (sourced by every driver), bench drivers, a job runner
    tests/      bitcheck_train.py, score_regress.py

## Data and components not included

The VULCAN runs, caches, truth sets, benches and checkpoints are not in this repository. `lib/paths.py` locates them
from two environment variables, `CHEM_PROJECT` and `CHEM_SCRATCH` (defaults are those of the machine the model was
trained on; `run/env.sh` exports them). The cache builder (`lib/rawtraj.py`) also imports `processing/preprocessing.py`
of the 12-species code at commit 7d5b036, checked out at `paths.SNAPSHOT`, whose `normalization.json` is the default
12-species normalization.

Two components are separate and not included, because they are GPL-licensed: the training-data generator (a 0-D wrapper
of VULCAN) and mini-chem. This repository only reads their output files.

## Requirements

Linux with a CUDA GPU (training and benches use CUDA graphs and `torch.compile`). Python 3.12.14 (`.python-version`):

    uv venv --python 3.12.14 && uv pip sync requirements.lock

`requirements.lock` pins the environment the model was trained and benched with (torch 2.8.0+cu126, numpy, h5py; it
also holds packages this code no longer imports, such as Lightning and matplotlib).

## License and citation

MIT (`LICENSE`). Citation metadata: `CITATION.cff`.
