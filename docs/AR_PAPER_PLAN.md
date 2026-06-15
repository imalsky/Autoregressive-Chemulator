# Auto-Chem ApJ paper — plan, outline & evidence map

**Working title:** *Stable Long-Horizon Autoregressive Emulation of Stiff Chemical Kinetics for Exoplanet Atmospheres*

**Type:** ApJ methods paper (aastex701, twocolumn), companion/sequel to the submitted one-step paper *"Accelerating Chemical Kinetics for Exoplanet Atmospheres using Neural Networks"* (Malsky & Zhang et al.; `Chemulator_ApJ/main.tex`).

**One-sentence thesis:** The submitted one-step flow-map emulator degrades under autoregressive rollout (published: 14.5 % err @10 steps → 56.6 % @99); this paper presents the *training methodology* (two-stage one-jump→curriculum AR fine-tuning, plus architecture/loss choices validated on a stiff benchmark) that controls long-horizon error accumulation, enabling GCM-deployable chemical-kinetics emulation at chemistry cost no greater than the reduced classical solver (mini-chem).

> This is the explicitly promised sequel — the submitted paper says (≈ line 453): *"a GCM implementation will require training procedures designed specifically to control error accumulation over long roll-outs. In future work, we will adapt training for long-horizon autoregressive tasks."*

---

## Headline decisions (settled with author, 2026-06-15)

1. **Scope** = AR *training-methodology* paper, distinct from the one-step paper (no salami-slice).
2. **Headline = long-rollout stability**, not speed.
3. **Headline accuracy test = fixed-dt deployment rollout** (dt ∈ {0.5, 6.4, 100} s, K ≥ 100): **stage-1 (one-step) vs stage-2 (AR)**, same production data, same architecture. GCM-relevant; matches `docs/DECISIONS.md` acceptance gate.
4. **Speed = supporting.** Measured locally; mini-chem in-process baseline; VULCAN by sibling-paper framing (no local rebuild).
5. **Process** = interactive, staged. **Hybrid execution:** draft everything not gated on HPC now; run the AR campaign in parallel; drop in the headline figure when it lands.

---

## Section outline

1. **Abstract** — stability problem, two-stage AR recipe, fixed-dt rollout result (stage-1→stage-2), speed comparable to mini-chem / GPU-batched, deployment.
2. **Introduction** — GCM chemistry cost; one-step emulator + its long-rollout failure (cite sibling); why AR/long-horizon is the open problem; contribution. Related work: ML chemistry emulators (Huang 2022, Hendrix 2023, Vojtekova 2025, etc., from the shared `references.bib`) + AR forecasting/curriculum literature (GraphCast, Stormer, FourCastNet, pushforward/Brandstetter 2022). [Pipeline Stage 1 — reuse sibling bib]
3. **Methods**
   - 3.1 Problem: flow map `y_{t+Δt}=Φ(y_t,g,Δt)`; autoregressive rollout; deployment loop.
   - 3.2 Data: VULCAN (52 sp / ~1200 rxn), 12 tracked species, (P,T), dt range, preprocessing → normalized shards. (Summarize; shared with sibling — cite.)
   - 3.3 Architecture: residual MLP (10×2048, SiLU, LayerNorm, predict-delta); latent-linear variant (Robertson).
   - 3.4 **Two-stage training (core contribution):** stage-1 one-jump pretrain → stage-2 AR fine-tune (lr/10); curriculum K=1→10; pushforward skip; BPTT vs detached + γ; EMA. (Evidence: `docs/DECISIONS.md`.)
   - 3.5 Loss: hybrid log10-MAE + z-MSE.
4. **Results**
   - 4.1 **Robertson sandbox** (architecture + training ablations): latent-linear 3–5× vs MLP; scale-dependent Fourier-Δt reversal; K-curriculum inverted-U; input-noise failure; single>banded; two-stage benefit. [MEASURED]
   - 4.2 **Production fixed-dt rollout stability** (HEADLINE): error-vs-step curve, stage-1 vs stage-2, dt ∈ {0.5,6.4,100} s, K≥100; per-species + conservation drift. [HPC-PENDING]
   - 4.3 **Speed:** CPU/MPS throughput vs mini-chem (in-process); comparable on commodity HW; orders-of-magnitude on datacenter GPU (sibling A100). [MEASURED local]
5. **Discussion & Conclusions** — Canoe deployment, limitations (transfer caveats, recipe provenance), future work.

---

## Evidence map (honesty ledger)

| Claim / result | Status | Backing artifact |
|---|---|---|
| One-step model degrades under rollout (the motivating failure) | Published | sibling `main.tex` (14.5 %@10, 56.6 %@99) |
| Latent-linear arch 3–5× better than MLP (stiff benchmark) | MEASURED | `Robertson_Tests/autoregressive/LL_CAMPAIGN.md`, `figures_l1/` |
| Fourier-Δt scale-reversal; K-curriculum; input-noise fails; single>banded; two-stage | MEASURED | `EXPERIMENTS.md`, `notes.md`, `docs/DECISIONS.md` |
| Architecture + two-stage recipe (definition) | KNOWN | `configs/stage1.json`, `stage2.json`, `docs/DECISIONS.md` |
| Speed: mini-chem cold-start 92 µs (eq) / 178 µs (off-eq) per cell, in-process | MEASURED | `mini_chem/src_mini_chem_dlsode/bench_dlsode.f90` |
| Speed: emulator CPU ~71 / MPS ~20 µs/cell (batched); comparable to mini-chem on commodity HW | MEASURED | `Auto-Chem/testing/bench_eager_devices.py`, `models/v3/bench_speedup.csv` |
| Speed: orders-of-magnitude vs VULCAN / on A100 | Sibling framing | sibling `main.tex` Fig 2 |
| Rollout eval harness (percentile vs step) works | VALIDATED | `Auto-Chem/testing/eval_rollout_error.py` (models/0: 100% finite, 5.7%@98 in easy regime) |
| **Production fixed-dt rollout: stage-1 vs stage-2 (HEADLINE)** | **HPC-PENDING** | run campaign below |

⚠️ **Caveat to disclose:** the shipped stage-2 recipe (BPTT + γ=1.0 + skip-1 + curriculum 2→10) was assembled from separate single-seed experiments and not validated end-to-end; the replicated combo is the detached+γ=0.9+skip-2 fallback. The HPC run is its first end-to-end validation — if it misbehaves, flip to the fallback.

---

## Critical path: HPC AR campaign → headline figure

Everything else in the paper is writable now. The headline (4.2) needs:

1. **Raw VULCAN HDF5** present at `configs/stage1.json:paths.raw_dir` on the cluster.
2. `qsub run.pbs` — preprocess (once) + **stage-1 one-jump pretrain** → `models/stage1/` (this checkpoint = the **one-step baseline**).
3. `qsub -W depend=afterok:<stage1_jobid> run_stage2.pbs` — **stage-2 AR fine-tune** → `models/stage2/` (the **AR model**).
4. **Export both** for inference: run `testing/export.py` with `RUN_DIR` = stage-1 work_dir, then = stage-2 work_dir (`EXPORT_DEVICES="cpu,cuda"`).
5. **Fixed-dt eval set (NEW, small task):** add a constant-dt eval-set generator (interpolate the same raw VULCAN trajectories onto t = k·dt grids for dt ∈ {0.5, 6.4, 100} s, K ≥ 100) → fixed-dt test shards.
6. **Headline figure:** run `testing/eval_rollout_error.py <stage1_export>` and `<stage2_export>` against the fixed-dt eval shards → two `rollout_error_curve.csv` → plot error-vs-step (stage-1 vs stage-2) at each dt. Target: stage-2 90th-pct fractional error ≪ stage-1, well under 10 %.

---

## Local benchmark tooling added for this paper (additive; no upstream solver edits)

- `mini_chem/src_mini_chem_dlsode/bench_dlsode.f90` (+ `mini_chem/bench.nml`, `bench_perturb.nml`, binary `mini_chem/bench_dlsode`) — in-process per-cell dlsode timer.
- `Auto-Chem/testing/bench_speedup.py` — exported-model batch latency sweep (CPU/MPS).
- `Auto-Chem/testing/bench_eager_devices.py` — eager CPU vs MPS latency (GPU number).
- `Auto-Chem/testing/eval_rollout_error.py` — percentile rollout-error vs step (reused for stage-1/stage-2).
