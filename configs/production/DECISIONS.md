# Production config decisions — evidence and justification

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

Date: 2026-06-09 (revised 2026-06-11). Scope: the autoregressive production configs in this directory
(`stage1_dt_1e-1_1e3.json` + `stage2_dt_1e-1_1e3.json`, plus `fallback_banded/`),
the two src changes that support them (`model.fourier_dt`, `loss_discount_gamma`),
and the validation protocol. Campaign log with full numbers:
`Chemulator_Project/notes.md`. Experiment artifacts:
`Chemulator_Project/Robertson_Tests/autoregressive/runs_e*`.

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
   — a strict superset of the legacy input. Wired into both FlowMapMLP and the
   autoencoder's LatentDynamics. torch.export-verified.
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
| stage-1 `lr` / `wd` / scheduler / EMA 0.9995 / early-stop | template values | v3-lineage; converged to val ~7e-4 historically. EMA window rule (1/(1−d) ≈ 1–10% of updates, ACE uses 0.9999) — with full-dataset epochs, consider 0.9999; not locally discriminable. | inherited |
| **stage-2 model block = stage-1 model block** | 10×2048 + fourier_dt | Fixes the historical inconsistency: old `configs/stage2_dt_bands/` declared 8×1536 but strict-load 10×2048 checkpoints from wrong paths — would hard-fail at startup. | high (correctness fix) |
| `training.rollout_steps` (stage 2) | 10 | **E3 K_MAX sweep**: inverted-U on geometric — K=1: 6.5e-3–1.9e-2, K=5: 4.4e-3, K=10: 2.4e-3 (±3%, 3 seeds), K=20: 1.2–3.3e-2 + one 1.0-dex fixed-dt blow-up. Matches List et al. 2024 (detached unrolling counterproductive beyond ~6; BPTT helps further but K=10 is inside every published envelope — GraphCast 12 max). Training horizon ≪ deployment horizon is universal practice. | medium-high |
| `curriculum` 2→10, linear, 30 epochs | enabled | All multi-step SOTA use staged/ramped horizons (GraphCast +1/1k-updates 1→12; Stormer 1→4→8; NeuralGCM "critical"); sandbox used the equivalent ramp in every successful run. | high (literature + used throughout locally) |
| `autoregressive_training.skip_steps` | 1 | Brandstetter 2022 canonical pushforward (unroll 2, backprop last; "more stable"); composes with BPTT. Sandbox used skip 2 at K≥4 successfully; 1 is the literature value and wastes fewer target steps. | medium |
| `autoregressive_training.detach_between_steps` | **false (full BPTT)** | **E4**: BPTT ≈ detach locally (geometric 2.6–3.0e-3 vs 2.4e-3; fixed-dt 0.30–0.37 vs 0.37–0.55 — within noise, both seeds). Tie broken by literature: every SOTA AR forecaster backprops through the rollout (GraphCast, Keisler, FourCastNet, Stormer, NeuralGCM, ACE2); the only *chemistry* precedent (Kelp 2020, 24-step recurrent) is full BPTT; List 2024: BPTT 38% vs detached 23%. Memory at K=10/10×2048/batch-2048 is fine on GH200. | medium (local tie; high literature prior) |
| `autoregressive_training.loss_discount_gamma` | 1.0 (with BPTT) | **E3+E4 interaction**: under BPTT, γ=1.0 ≈ γ=0.9 (3.3e-3 vs 2.6e-3, within noise); uniform matches all SOTA practice. **Under DETACH, γ<1 is required**: uniform was 4–14× worse in both seeds (1.1e-2, 3.3e-2 vs 2.4e-3 at γ=0.9). Hence fallback recipe pins γ=0.9. | medium-high |
| `backward_per_step` | false | Only meaningful in the detached path; BPTT does one backward. Fallback recipe sets true (memory). | n/a |
| stage-2 `lr` | 3e-5 (= stage-1 peak / 10) | **E5**: staged pretrain→fine-tune at LR peak/10 gave the best fixed-dt result of the whole campaign (0.256 vs 0.379 from-scratch); LR peak/3 (1e-4) doubled geometric error with no fixed-dt gain. Literature: AR fine-tune LR is 5–3000× below pretrain peak (FourCastNet /5, Keisler /10 per stage, Stormer /100–/1000, GraphCast /3000); /10 is at the aggressive-but-precedented end, and the old template's 2e-5 is also fine. | medium-high |
| stage-2 `wd` | 1e-5 (= stage 1; old template's 0.005 dropped) | No SOTA fine-tune cranks WD ×500 at the phase boundary; the 0.005 value was untested (stage 2 never ran). Keep optimizer geometry constant except LR. | medium (literature/consistency) |
| stage-2 `lambda_z_mse` 0.25, EMA 0.999, early-stop 20, plateau warmup 5 | template values | Inherited from the stage-2 template; not locally discriminated. EMA re-warms automatically at stage start (fresh callback state); ACE evidence says EMA helps long-rollout metrics up to 15%. | inherited |
| **rejected: input-noise injection** | not shipped | **E6**: 4/4 noise runs worse, 3/4 catastrophic at fixed-dt (12.4–14.1 dex; σ ∈ {0.01, 0.03} z-space, with and without pushforward, 2 seeds) — *against* the GNS/Stachenfeld prior. No trainer option added. If ever revisited, sweep σ ≤ 1e-3 and re-test. | high (locally decisive) |
| **rejected: K=20 detached training** | — | E3: consistently worse + one 1-dex blow-up; List 2024 agrees. (K>10 with BPTT untested locally — plausible future direction, not a default.) | medium |
| **not adopted: inference-time renormalization** | off | Sturm 2024: naive projection degrades radicals/trace species (exactly what dominates a log-standardized loss); weighted projection neutral-to-positive. Track conservation drift as an EVAL metric; if the host solver needs hard conservation, the stoichiometric-output-layer route (Sturm & Wexler 2022) is the accuracy-neutral fix. | high (literature) |

## Fallbacks shipped

1. **`fallback_banded/`** — two-band layout (stage1+stage2 × {1e-1–1e1, 1e1–1e3} s).
   Use only if the single model shows a measured gap at the deployment dt that a
   deployment-dt-concentrated stage-2 (below) cannot close.
2. **Detached stage-2 recipe** (lower memory, equally validated locally): in
   `stage2_dt_1e-1_1e3.json` set `detach_between_steps=true`, `backward_per_step=true`,
   `loss_discount_gamma=0.9`. Do NOT run detached with γ=1.0 (E3: 4–14× worse).
3. **Deployment-specialized fine-tune** (if production rollout at dt≈6.4 s underperforms):
   rather than banding, run a second stage-2 pass with preprocessing dt_min/dt_max
   narrowed to ~[2, 20] s from the *same* stage-1 checkpoint (FuXi-style). Keeps one
   model family, no band-boundary discontinuity.
4. **Autoencoder trunk** — `model.type=autoencoder` remains supported (incl. fourier_dt);
   Fourier-Flow-map was worse + 5.5× seed spread at sandbox scale (E1) but small-scale
   bottleneck behavior may not transfer; documented, not pursued.

## Cheap production-scale spot-checks (optional, per the transfer-risk policy)

Each ~20 stage-1 epochs (~5% of a full run) on one GH200, comparing val curves:
- **fourier_dt on/off** — the single most load-bearing adoption.
- **fourier_dt.sigma ∈ {0.5, 1, 2}** — only if the on/off check shows sensitivity.
- **stage-2, first 10 epochs**: BPTT+γ=1.0 vs detached+γ=0.9 — verifies the E4 tie at scale
  before committing the remaining 190 epochs.

## Verification performed (2026-06-09)

- **End-to-end smoke** (`run_local_smoke.sh`): synthetic raw → preprocessing → stage-1
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
  dynamic batch at fp32 and bf16. Review findings addressed: fallback_banded path depth
  fixed (+ resolution verified); `fourier_dt` block without `enabled` now raises;
  `loss_discount_gamma` outside AR mode now raises; Fourier projection computed in fp32
  internally (bf16-config fidelity). Known non-blocking notes: val_loss is intentionally
  uniform-weighted even when training γ<1 (cross-config comparability — commented in
  trainer.py); AR + bf16-mixed + fused-optimizer + clipping fails at first step
  (pre-existing Lightning guard; production configs are fp32-true and unaffected).
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
