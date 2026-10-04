#!/usr/bin/env python3
"""Rollout-error curves of an exported 1-step model over a seeded sample of one split (diagnostic).

Rolls the run's exported CPU model (export_cpu_dynB_1step_dlog10.pt2 in the run directory named by
AUTOCHEM_CONFIG_PATH, testing/run_dir.py) autoregressively at each window's own fixed dt over a
reproducible sample of EVAL_SPLIT: EVAL_N_SOURCES source trajectories drawn with EVAL_SEED, one
seeded window per source. Each step applies the export's log10 increment to a predictions.HOST_DTYPE
host state with testing/export.py host_step. The windows are the processed random-dt shards (dt
log-uniform in the trained range), so K <= T - 1 - ROLLOUT_START_INDEX (98 today) and every row is
labelled "random-dt, K <= 98": no fixed-dt evaluation set exists yet.

Reported per rollout step, for the model and for persistence (a zero increment through host_step:
y_next = max(y, epsilon), the start state carried forward):
- per-species |log10(yp) - log10(yt)| p50 / p90 in dex (recommended headline; the author confirms
  the paper target) and the median / mean signed log10(yp / yt) (bias); 10% fractional error is
  about 0.041 dex for an overshoot and 0.046 dex for an undershoot. The headline states its rollout
  step and its dt (each window's own constant dt; there is no fixed-dt set yet);
- the legacy pooled fractional-error percentiles |yp - yt| / (|yt| + FRAC_ERR_FLOOR) for continuity
  (the floor hides trace-species error; the share of scored truth below 1e-12 and 1e-20 is printed);
- n_finite and n_refused: host_step keeps a row's previous state when its update is not finite or
  leaves the log10 y envelope, so a diverging row stays finite and is counted in n_refused (rows whose
  update was refused at that step) instead. Percentiles keep a fixed population: a non-finite entry
  ranks as +inf error.
Strata: a window is steady state when its truth stays within STEADY_STATE_MAX_DEX of its start
state at every step and species (persistence is nearly exact there); the others are evolving. The
headline rows and the pooled dex p90 at the report steps are also reported for each stratum, the
evolving rows being the recommended headline, so steady states do not dominate it; the curve CSVs
pool all windows. The loss against persistence and gate 1 are judged on the evolving windows only
(a ratio against a nearly exact persistence is meaningless); the steady-state windows are reported
apart by the model's absolute log10 error.
Loss against persistence: the run's val_loss definition (training.rollout_steps from the rollout
start, steps >= skip_steps for an AR run, the run's loss weights), recomputed in float64 from the
host states (values clipped at epsilon), for the model and for persistence, as a ratio
of means and skill = 1 - ratio, overall and by window-start decade, over the evolving windows (the
all-window pooled ratio, training's val_skill population, is printed as a diagnostic). The
window-start time is t_start + ROLLOUT_START_INDEX * dt; the shards hold no time, so t_start is
replayed from the run's raw files with preprocessing's own per-trajectory draws and every replayed
window is checked against its shard row. Gate 1 (the stage-1 accuracy gate, PROVISIONAL; GATE1_*
below; evolving windows only) is printed as pass/fail lines for a one-jump run; it is reported only,
never enforced (the exit status ignores it).
Bootstrap CIs over sources are given at the report steps. Sources whose dt, P or T lies outside the
export's trained box (NaN rows by the export contract) are dropped before the rollout and counted
apart from divergence. If this run was started with weights_only from another run whose evaluation
on the same selection exists, a paired bootstrap of the difference is added.

Run (once per stage, e.g. after export.py):
  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/eval_rollout_error.py
Outputs (in the run directory, named by split): rollout_error_curve_<split>.csv (model and
persistence columns), rollout_error_species_<split>.csv, rollout_error_<split>.json (provenance,
CIs, loss against persistence and the gate-1 lines) and rollout_error_<split>_per_source.npz
(per-source errors at the report steps, window-start times and losses, for paired comparisons).
The CPU export runs at full float32 (TF32 does not apply on CPU). Inputs besides the export and the
shards: the build's raw files (window-start times), as recorded in its preprocessing_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))
sys.path.insert(0, str(ROOT / "processing"))

import export as E  # noqa: E402  (export's baked normalizer: the round-trip check; host_step; src/utils fingerprint)
import predictions as P  # noqa: E402  (its independent numpy inverse normalization and loaders)
import preprocessing as PP  # noqa: E402  (its own window draws: the window-start time replay)
from run_dir import resolve_run_dir  # noqa: E402
from utils import file_sha256, lock_processed_dir_shared, normalization_sha256  # noqa: E402

# Every gate, flip and trigger decision uses validation; 'test' only for the single final paper run.
EVAL_SPLIT = "validation"
# None => the run config's paths.processed_dir; or the root of a designed evaluation set in the
# same shard schema and normalization (checked by norm_sha256), built by preprocessing.py: the
# window-start replay reads its preprocessing_summary.json and the raw files it records.
EVAL_SHARD_ROOT: str | None = None
EVAL_N_SOURCES = 4000          # inherited value; one window per source
EVAL_SEED = 0                  # source and window selection
EVAL_BOOTSTRAP_N = 1000        # resamples of the sources for the CIs
EVAL_BOOTSTRAP_SEED = 0
EVAL_CI_LEVEL = 0.95
ROLLOUT_START_INDEX = P.ROLLOUT_START_INDEX  # inherited (index 0 skipped; see predictions.py)
PCTLS = (10, 50, 90, 95)       # legacy pooled fractional percentiles
SPECIES_PCTLS = (50, 90)       # per-species dex percentiles
# The sibling one-step paper reported @10 and @99 steps; targets above K are clamped to K and each
# row is labelled with the step actually reported.
HEADLINE_STEPS = (10, 99)
REPORT_STEPS = (1, 5, 10, 25, 50, 99)
FRAC_ERR_FLOOR = P.FRAC_ERR_FLOOR
TRUTH_DISCLOSURE_LEVELS = (1e-12, 1e-20)
# Physical inputs rebuilt from the shards must re-normalize (export's baked normalizer, float32) to
# the shard values within this z / dt_norm distance: about 20x the measured float32 round trip
# (4.8e-7); a wrong statistic shows up as a much larger error.
NORM_ROUNDTRIP_TOL_Z = 1e-5
DATA_LABEL = "random-dt, K <= 98 (processed shards; no fixed-dt evaluation set yet)"
# Gate 1, the stage-1 accuracy gate (PROVISIONAL; revisit after the first trained model): skill
# against persistence, because persistence already scores a stage-1 validation loss of ~1.1e-2 and
# its loss falls ~1000x from early to late window starts.
# Pass, on the evolving windows only (steady-state windows, where persistence is nearly exact, are
# reported apart by absolute log10 error): overall loss <= GATE1_MAX_LOSS_RATIO x persistence, and
# loss < persistence in every window-start decade that has evolving windows (ratio <
# GATE1_MAX_DECADE_RATIO); the decades from GATE1_LATE_START_S on also get their own line.
GATE1_MAX_LOSS_RATIO = 0.5
GATE1_MAX_DECADE_RATIO = 1.0
GATE1_LATE_START_S = 1e3
# Steady-state stratum: a window whose truth stays within this of its start state (|dlog10|, at every
# step and species) is steady state, where persistence is (nearly) exact; it is reported apart from
# the evolving windows. The same threshold preprocessing cuts frozen raw tails at.
STEADY_STATE_MAX_DEX = PP._FROZEN_TAIL_MAX_PRECHANGE_DEX
# Window-start replay: a shard row matches a raw group by (P, T) and a replayed draw by dt within
# this relative tolerance (float32 z and dt_norm storage moves them by ~1e-6), and the replayed
# window must re-normalize to the shard row within NORM_ROUNDTRIP_TOL_Z.
REPLAY_MATCH_RTOL = 1e-5


def _pctl_fixed_pool(x: np.ndarray, q: Any, axis: Any = None) -> np.ndarray:
    """Percentiles over the whole pool; a non-finite (diverged) entry ranks as +inf error."""
    x = np.where(np.isfinite(x), x, np.inf)
    with np.errstate(invalid="ignore"):
        ps = np.percentile(x, q, axis=axis)  # linear estimator; NaN only where interpolation touches +inf
    return np.where(np.isnan(ps), np.percentile(x, q, axis=axis, method="higher"), ps)


def _source_keys(globals_z: np.ndarray) -> np.ndarray:
    """One uint64 key per row from its float32 (P, T) pair (each source has one unique pair)."""
    u = np.ascontiguousarray(globals_z, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (u[:, 0] << np.uint64(32)) | u[:, 1]


def _select_rows(shards: List[Path], *, max_rows_per_source: int) -> tuple[list[tuple[int, int]], int, int]:
    """Pass 1 (globals only): draw EVAL_N_SOURCES sources with EVAL_SEED and one window each.

    Returns the sorted (shard index, row) list, the number of sources and the number of rows.
    """
    keys, where = [], []
    for si, path in enumerate(shards):
        with np.load(path) as z:
            g = z["globals"]
        keys.append(_source_keys(g))
        where.append(np.column_stack([np.full(g.shape[0], si), np.arange(g.shape[0])]))
    allk = np.concatenate(keys)
    allw = np.concatenate(where)
    order = np.argsort(allk, kind="stable")
    allk, allw = allk[order], allw[order]
    uniq, start, count = np.unique(allk, return_index=True, return_counts=True)
    if int(count.max()) > max_rows_per_source:
        raise ValueError(
            f"a (P, T) pair holds {int(count.max())} rows, more than samples_per_source_trajectory="
            f"{max_rows_per_source}: (P, T) does not identify a source in this data"
        )
    rng = np.random.default_rng(EVAL_SEED)
    n = min(EVAL_N_SOURCES, uniq.size)
    picked = np.sort(rng.choice(uniq.size, size=n, replace=False))
    rows = [tuple(int(v) for v in allw[start[i] + int(rng.integers(0, count[i]))]) for i in picked]
    return sorted(rows), int(uniq.size), int(allk.size)


def _load_rows(shards: List[Path], rows: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 2: load only the selected rows (y_mat, globals, dt_norm_mat), one shard at a time."""
    ys, gs, dts = [], [], []
    by_shard: Dict[int, List[int]] = {}
    for si, r in rows:
        by_shard.setdefault(si, []).append(r)
    for si in sorted(by_shard):
        idx = np.asarray(by_shard[si])
        y, g, dt = P._load_shard(shards[si])
        ys.append(y[idx])
        gs.append(g[idx])
        dts.append(dt[idx])
    return np.concatenate(ys), np.concatenate(gs), np.concatenate(dts)


def _check_roundtrip(manifest, species_keys, global_keys, *, y0_phys, g_phys, dt_seconds, y0_z, g_z, dt_norm) -> None:
    """The physical inputs must re-normalize to the shard values through the export's baked normalizer."""
    norm = E.build_baked_normalizer(manifest, species_vars=species_keys, global_vars=global_keys)
    with torch.no_grad():
        got = {
            "species z": norm.normalize_species(torch.from_numpy(y0_phys.astype(np.float32))).numpy(),
            "globals": norm.normalize_globals(torch.from_numpy(g_phys.astype(np.float32))).numpy(),
            "dt_norm": norm.normalize_dt_seconds(torch.from_numpy(dt_seconds.astype(np.float32))).numpy(),
        }
    want = {"species z": y0_z, "globals": g_z, "dt_norm": dt_norm}
    errs = {k: float(np.max(np.abs(got[k].astype(np.float64) - want[k].astype(np.float64)))) for k in got}
    if any(e > NORM_ROUNDTRIP_TOL_Z for e in errs.values()):
        raise ValueError(f"normalization round trip exceeds {NORM_ROUNDTRIP_TOL_Z:g}: max error per key {errs}")
    print(f"[check] normalization round trip max error per key: {errs}")


def _window_start_times(pcfg, summary, manifest, *, g_phys, dt_s, y_z) -> np.ndarray:
    """t_start (s, the window's first state) of each selected window, replayed from the run's raw files.

    The shards hold no time. Each window's source group is found by its (P, T) among the EVAL_SPLIT
    groups of the build's raw files, its draws are replayed with preprocessing's own per-trajectory
    generator (seed, group name) and functions as in sample_file (valid = t_raw > 0, cut at a frozen
    raw tail by preprocessing.frozen_tail_cut), the draw is matched by dt, and the replayed window
    must re-normalize to the shard row (z) within NORM_ROUNDTRIP_TOL_Z. Any miss raises, as does a
    build whose recorded frozen-tail threshold is not this code's. g_phys [N, G], dt_s [N] and
    y_z [N, T, S] are the rows'.
    """
    recorded = (summary.get("frozen_tail") or {}).get("max_prechange_dex")
    if recorded != PP._FROZEN_TAIL_MAX_PRECHANGE_DEX:
        raise ValueError(f"the build records frozen_tail.max_prechange_dex={recorded!r}, not this code's "
                         f"{PP._FROZEN_TAIL_MAX_PRECHANGE_DEX!r}: its windows cannot be replayed; rebuild it")
    names, files, gs = [], [], []
    for r in summary["raw_files"]:
        path = pcfg.raw_dir / r["name"]
        if not path.is_file() or path.stat().st_size != int(r["size_bytes"]):
            raise FileNotFoundError(f"{path}: missing or not the size the build recorded ({r['size_bytes']} bytes)")
        with h5py.File(path, "r") as fin:
            for name in fin.keys():
                if PP.split_for_trajectory(r["name"], name, seed=PP.SPLIT_SEED, val_frac=pcfg.val_fraction,
                                           test_frac=pcfg.test_fraction) != EVAL_SPLIT:
                    continue
                grp = fin[name]
                try:
                    g = PP.read_globals_vector(grp, global_vars=pcfg.global_variables, leaf_index=PP.leaf_dataset_index(grp))
                except (KeyError, TypeError, ValueError):
                    continue  # a group preprocessing rejects at its globals has no windows
                names.append(name)
                files.append(path)
                gs.append(g)
    raw_g = np.asarray(gs, dtype=np.float64)
    order = np.argsort(raw_g[:, 0], kind="stable")
    raw_g = raw_g[order]
    mu = np.array([float(manifest["per_key_stats"][s]["log_mean"]) for s in pcfg.species_variables])
    sd = np.array([float(manifest["per_key_stats"][s]["log_std"]) for s in pcfg.species_variables])
    by_file: Dict[Path, List[tuple[int, str]]] = {}
    for i in range(g_phys.shape[0]):
        lo = np.searchsorted(raw_g[:, 0], g_phys[i, 0] * (1.0 - REPLAY_MATCH_RTOL), side="left")
        hi = np.searchsorted(raw_g[:, 0], g_phys[i, 0] * (1.0 + REPLAY_MATCH_RTOL), side="right")
        hit = [j for j in range(lo, hi) if abs(raw_g[j, 1] / g_phys[i, 1] - 1.0) <= REPLAY_MATCH_RTOL]
        if len(hit) != 1:
            raise ValueError(f"window {i}: (P, T) = {g_phys[i].tolist()} matches {len(hit)} raw {EVAL_SPLIT} groups, not 1")
        j = int(order[hit[0]])
        by_file.setdefault(files[j], []).append((i, names[j]))
    t_start = np.full(g_phys.shape[0], np.nan)
    offsets = np.arange(pcfg.n_steps, dtype=np.float64)
    for path, items in by_file.items():
        with h5py.File(path, "r") as fin:
            for i, name in items:
                grp = fin[name]
                li = PP.leaf_dataset_index(grp)
                t_raw = PP.read_time(grp, time_key=pcfg.time_key, leaf_index=li)
                y_raw = PP.read_species_matrix(grp, t_len=t_raw.shape[0], species_vars=pcfg.species_variables, leaf_index=li)
                valid = t_raw > 0
                cut = PP.frozen_tail_cut(y_raw, epsilon=pcfg.epsilon)
                if cut is not None:
                    valid &= np.arange(t_raw.shape[0]) <= cut
                t_valid, y_valid = t_raw[valid], y_raw[valid, :]
                rng = np.random.default_rng([int(pcfg.seed), PP._seed_word(str(name))])
                draws = []
                for sidx in range(pcfg.samples_per_source_trajectory):
                    d = PP.sample_dt(pcfg, rng)
                    t0 = PP.pick_t_start(t_raw=t_raw, t_valid=t_valid, dt_s=d, n_steps=pcfg.n_steps, t_min=pcfg.t_min,
                                         rng=rng, anchor_first=(sidx == 0), max_attempts=pcfg.max_chunk_attempts_per_source)
                    if t0 is not None and abs(d / dt_s[i] - 1.0) <= REPLAY_MATCH_RTOL:
                        draws.append((d, t0))
                if len(draws) != 1:
                    raise ValueError(f"window {i} ({path.name}:{name}): {len(draws)} replayed draws match dt={dt_s[i]:.9g} s, not 1")
                d, t0 = draws[0]
                i0, i1, w = PP.prepare_log_interp(np.log10(t_valid), np.log10(t0 + offsets * d))
                y_new = PP.interp_loglog(y_valid, i0=i0, i1=i1, w=w, epsilon=pcfg.epsilon).astype(np.float32)
                z = (np.log10(np.maximum(y_new.astype(np.float64), pcfg.epsilon)) - mu) / sd
                err = float(np.max(np.abs(z - y_z[i].astype(np.float64))))
                if err > NORM_ROUNDTRIP_TOL_Z:
                    raise ValueError(f"window {i} ({path.name}:{name}): the replayed window differs from its shard row by {err:.3g} (z)")
                t_start[i] = t0
    return t_start


def _loss_definition(cfg: Dict[str, Any], manifest: Dict[str, Any], species_keys: List[str]) -> Dict[str, Any]:
    """The run's val_loss definition: training.rollout_steps steps, steps >= skip_steps weighted 1 in an AR run."""
    t = cfg["training"]
    k = int(t["rollout_steps"])
    skip = int(t["autoregressive_training"]["skip_steps"]) if t["autoregressive_training"]["enabled"] else 0
    return {
        "k": k, "skip": skip, "weights": np.array([0.0 if i < skip else 1.0 for i in range(k)]),
        "lambda_log10_mae": float(t["loss"]["lambda_log10_mae"]), "lambda_z_mse": float(t["loss"]["lambda_z_mse"]),
        "log_std": np.array([float(manifest["per_key_stats"][s]["log_std"]) for s in species_keys]),
    }


@torch.inference_mode()
def _rollout_stats(model, y0_phys, g_phys, dt_seconds, y_true, *, eps: float, b_max: int, report_idx: List[int],
                   loss_def: Dict[str, Any]):
    """Steps in the outer loop, row chunks of at most b_max in the inner loop; keep only the [N, S] state
    (P.HOST_DTYPE, advanced by E.host_step).

    Returns per-step pooled frac percentiles, pooled dex p90, per-species dex percentiles and bias,
    n_finite, n_refused, the per-source refusal flags (sources_refused: any step refused), the
    truth-below-level shares, the per-source dex at the report steps, the per-source loss under loss_def
    (lambda_log10_mae * mean |dlog10| + lambda_z_mse * mean (dlog10 / log_std)^2 over species, weighted
    over the first loss_def["k"] steps; +inf for a non-finite source) and the per-source absolute log10
    error (mean |dlog10| over species, same step weights; +inf if non-finite).
    """
    N, K, S = y_true.shape
    state = torch.from_numpy(y0_phys).to(P.HOST_DTYPE)
    g = torch.from_numpy(g_phys.astype(np.float32))
    dt = torch.from_numpy(dt_seconds.astype(np.float32))
    out: Dict[str, Any] = {"frac": [], "dex_p90": [], "species_dex": [], "bias_median": [], "bias_mean": [],
                           "n_finite": [], "n_refused": [], "dex_at_report": []}
    refused_any = np.zeros(N, dtype=bool)
    window_loss = np.zeros(N)
    window_abs_dex = np.zeros(N)
    below = {lvl: 0 for lvl in TRUTH_DISCLOSURE_LEVELS}
    for k in range(K):
        nxt = [E.host_step(model, state[i:i + b_max], dt[i:i + b_max, k], g[i:i + b_max], epsilon=eps)
               for i in range(0, N, b_max)]
        state = torch.cat([s for s, _ in nxt], dim=0)
        refused = torch.cat([r for _, r in nxt], dim=0).numpy()
        refused_any |= refused
        out["n_refused"].append(int(refused.sum()))
        yp = state.to(torch.float64).numpy()
        yt = y_true[:, k, :]
        for lvl in below:
            below[lvl] += int((yt < lvl).sum())
        ypc, ytc = np.clip(yp, eps, None), np.clip(yt, eps, None)
        frac = np.abs(ypc - ytc) / (np.abs(ytc) + FRAC_ERR_FLOOR) * 100.0
        with np.errstate(invalid="ignore", divide="ignore"):
            signed = np.log10(ypc) - np.log10(ytc)
        dex = np.abs(signed)
        out["frac"].append(_pctl_fixed_pool(frac.ravel(), PCTLS))
        out["dex_p90"].append(float(_pctl_fixed_pool(dex.ravel(), 90)))
        out["species_dex"].append(_pctl_fixed_pool(dex, SPECIES_PCTLS, axis=0))  # [len(SPECIES_PCTLS), S]
        fin = np.isfinite(signed)
        with np.errstate(invalid="ignore"):
            out["bias_median"].append(np.nanmedian(np.where(fin, signed, np.nan), axis=0))
            out["bias_mean"].append(np.nanmean(np.where(fin, signed, np.nan), axis=0))
        out["n_finite"].append(int(np.isfinite(yp).all(axis=1).sum()))
        if k in report_idx:
            out["dex_at_report"].append(dex)
        if k < loss_def["k"] and loss_def["weights"][k] > 0.0:
            with np.errstate(invalid="ignore"):
                mae = dex.mean(axis=1)
                term = (loss_def["lambda_log10_mae"] * mae
                        + loss_def["lambda_z_mse"] * np.square(dex / loss_def["log_std"]).mean(axis=1))
            window_loss += loss_def["weights"][k] * np.where(np.isfinite(term), term, np.inf)
            window_abs_dex += loss_def["weights"][k] * np.where(np.isfinite(mae), mae, np.inf)
    out["sources_refused"] = refused_any
    out["truth_share_below"] = {f"{lvl:g}": below[lvl] / float(N * K * S) for lvl in below}
    out["window_loss"] = window_loss / loss_def["weights"].sum()
    out["window_abs_dex"] = window_abs_dex / loss_def["weights"].sum()
    return out


def _bootstrap_ci(dex_at_report: np.ndarray) -> Dict[str, Any]:
    """Percentile-bootstrap CIs over sources of the per-species dex p50/p90 and the pooled p90."""
    rng = np.random.default_rng(EVAL_BOOTSTRAP_SEED)
    R, N, S = dex_at_report.shape
    lo_q, hi_q = 50.0 * (1 - EVAL_CI_LEVEL), 50.0 * (1 + EVAL_CI_LEVEL)
    sp = np.empty((EVAL_BOOTSTRAP_N, R, len(SPECIES_PCTLS), S))
    pooled = np.empty((EVAL_BOOTSTRAP_N, R))
    for b in range(EVAL_BOOTSTRAP_N):
        idx = rng.integers(0, N, size=N)
        d = dex_at_report[:, idx, :]
        sp[b] = np.stack([_pctl_fixed_pool(d[r], SPECIES_PCTLS, axis=0) for r in range(R)])
        pooled[b] = [float(_pctl_fixed_pool(d[r].ravel(), 90)) for r in range(R)]
    return {
        "species_dex": {"lo": np.percentile(sp, lo_q, axis=0).tolist(), "hi": np.percentile(sp, hi_q, axis=0).tolist()},
        "pooled_dex_p90": {"lo": np.percentile(pooled, lo_q, axis=0).tolist(), "hi": np.percentile(pooled, hi_q, axis=0).tolist()},
    }


def _paired_difference(run_dir: Path, here: np.ndarray, selection_sha: str, name: str) -> Any:
    """Paired bootstrap (this run - its weights_only source run) of the per-species dex p90, if available."""
    cfg = P._load_json(run_dir / "config.resolved.json")
    if cfg["training"]["checkpoint_mode"] != "weights_only":
        return "not applicable (this run was not started from another run)"
    src = (run_dir / cfg["runtime"]["checkpoint"]).resolve().parent.parent
    other = src / name
    if not other.exists():
        return f"not available: evaluate the source run {src} first (same split and selection)"
    with np.load(other) as z:
        if str(z["selection_sha256"]) != selection_sha:
            return f"not comparable: {other} scored a different selection"
        there = z["dex_at_report"]
    rng = np.random.default_rng(EVAL_BOOTSTRAP_SEED)
    R, N, S = here.shape
    diffs = np.empty((EVAL_BOOTSTRAP_N, R, S))
    for b in range(EVAL_BOOTSTRAP_N):
        idx = rng.integers(0, N, size=N)
        diffs[b] = np.stack([_pctl_fixed_pool(here[r, idx], 90, axis=0) - _pctl_fixed_pool(there[r, idx], 90, axis=0)
                             for r in range(R)])
    lo_q, hi_q = 50.0 * (1 - EVAL_CI_LEVEL), 50.0 * (1 + EVAL_CI_LEVEL)
    point = np.stack([_pctl_fixed_pool(here[r], 90, axis=0) - _pctl_fixed_pool(there[r], 90, axis=0) for r in range(R)])
    return {"source_run": str(src), "per_species_dex_p90_difference": point.tolist(),
            "lo": np.percentile(diffs, lo_q, axis=0).tolist(), "hi": np.percentile(diffs, hi_q, axis=0).tolist()}


def main() -> None:
    run_dir = resolve_run_dir()
    export_path = P._resolve_export_path(run_dir)
    model, meta, infer_dtype = P._load_export_module(export_path)
    norm_path = P._resolve_normalization_path(meta)
    manifest = P._load_json(norm_path)
    if normalization_sha256(manifest) != meta["norm_sha256"]:
        raise ValueError(f"{norm_path} is not the normalization {export_path.name} was exported with (norm_sha256)")
    cfg = P._load_json(run_dir / "config.resolved.json")
    shard_root = (Path(EVAL_SHARD_ROOT).expanduser() if EVAL_SHARD_ROOT is not None
                  else (run_dir / cfg["paths"]["processed_dir"]).resolve())
    # Shared lock on the shard root from the manifest read until the rows are loaded (released below),
    # as training holds it: a preprocessing run rebuilding it holds it exclusively (src/utils.py).
    data_lock_fd = lock_processed_dir_shared(shard_root)
    root_manifest = P._load_json(shard_root / "normalization.json")
    if normalization_sha256(root_manifest) != meta["norm_sha256"]:
        raise ValueError(f"{shard_root} is not normalized like {export_path.name} (norm_sha256 differs)")
    summary = P._load_json(shard_root / "preprocessing_summary.json")
    species_keys = list(manifest["species_variables"])
    global_keys = list(manifest["global_variables"])
    eps = float(manifest["epsilon"])
    b_max = int(meta["dynamic_batch"]["max"])
    print(f"[model] {export_path}  (CPU float32 calls, TF32 does not apply; host state {P.HOST_DTYPE})")
    print(f"[data]  {shard_root}/{EVAL_SPLIT}  ({DATA_LABEL})")

    shards = sorted((shard_root / EVAL_SPLIT).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shard_root / EVAL_SPLIT}")
    rows, n_sources_total, n_rows_total = _select_rows(
        shards, max_rows_per_source=int(summary["config"]["samples_per_source_trajectory"]))
    selection_sha = hashlib.sha256(json.dumps([[shards[s].name, r] for s, r in rows]).encode()).hexdigest()
    y_mat, g_mat, dt_norm_mat = _load_rows(shards, rows)
    os.close(data_lock_fd)
    N, T, S = y_mat.shape
    K = (T - 1) - ROLLOUT_START_INDEX
    print(f"[eval]  {N} sources of {n_sources_total} ({n_rows_total} windows, {len(shards)} shards), seed {EVAL_SEED}; "
          f"T={T}, K={K} steps from index {ROLLOUT_START_INDEX}")

    y0_z = y_mat[:, ROLLOUT_START_INDEX, :]
    y_true_z = y_mat[:, ROLLOUT_START_INDEX + 1: ROLLOUT_START_INDEX + 1 + K, :]
    dt_seq_norm = dt_norm_mat[:, ROLLOUT_START_INDEX: ROLLOUT_START_INDEX + K]
    y0_phys = P._denormalize_species(y0_z, species_keys, manifest)
    y_true = P._denormalize_species(y_true_z, species_keys, manifest)
    g_phys = np.stack([P._denormalize_globals(g_mat[i], global_keys, manifest) for i in range(N)], axis=0)
    dt_seconds = P._dt_norm_to_seconds(dt_seq_norm, manifest)
    _check_roundtrip(manifest, species_keys, global_keys, y0_phys=y0_phys, g_phys=g_phys,
                     dt_seconds=dt_seconds[:, 0], y0_z=y0_z, g_z=g_mat, dt_norm=dt_seq_norm[:, 0])

    oor = np.zeros(N, dtype=bool)
    for k in range(K):
        oor |= E.input_out_of_range_rows(y0_phys, dt_seconds[:, k], g_phys, meta)
    if oor.any():
        print(f"[eval]  dropping {int(oor.sum())} of {N} sources outside the trained box (NaN rows by contract)")
    keep = np.logical_not(oor)
    y0_phys, y_true, g_phys, dt_seconds = y0_phys[keep], y_true[keep], g_phys[keep], dt_seconds[keep]
    kept_rows = [r for r, k_ in zip(rows, keep, strict=True) if k_]

    # The build's own preprocessing record (paths relative to its processed_dir) drives the replay.
    bc = summary["config"]
    pcfg = PP.PreCfg(**{**bc, "raw_dir": (shard_root / bc["raw_dir"]).resolve(),
                        "processed_dir": (shard_root / bc["processed_dir"]).resolve()})
    t_start = _window_start_times(pcfg, summary, manifest, g_phys=g_phys, dt_s=dt_seconds[:, 0], y_z=y_mat[keep])
    t_window = t_start + ROLLOUT_START_INDEX * dt_seconds[:, 0]
    print(f"[check] window-start times replayed from {pcfg.raw_dir} for {t_start.size} windows "
          f"(each re-interpolated window matches its shard row within {NORM_ROUNDTRIP_TOL_Z:g} in z)")

    report = sorted({min(s, K) for s in REPORT_STEPS})
    headline = sorted({min(s, K) for s in HEADLINE_STEPS})
    loss_def = _loss_definition(cfg, manifest, species_keys)
    if loss_def["k"] > K:
        raise ValueError(f"training.rollout_steps={loss_def['k']} exceeds the K={K} evaluated steps")
    stats = _rollout_stats(model, y0_phys, g_phys, dt_seconds, y_true, eps=eps, b_max=b_max,
                           report_idx=[s - 1 for s in report], loss_def=loss_def)
    pstats = _rollout_stats(lambda y, _dt, _g: torch.zeros_like(y), y0_phys, g_phys, dt_seconds, y_true, eps=eps, b_max=b_max,
                            report_idx=[s - 1 for s in report], loss_def=loss_def)
    n_kept = int(keep.sum())
    # Steady state: persistence's error stays within STEADY_STATE_MAX_DEX at every step and species.
    pers_dex = np.abs(np.log10(np.clip(y_true, eps, None)) - np.log10(np.clip(y0_phys, eps, None))[:, None, :])
    steady = pers_dex.max(axis=(1, 2)) <= STEADY_STATE_MAX_DEX
    strata = {"evolving": np.logical_not(steady), "steady_state": steady}
    n_steady = int(steady.sum())
    print(f"non-finite sources at the last step: {n_kept - stats['n_finite'][-1]}/{n_kept}; host_step refused the "
          f"update of {int(stats['sources_refused'].sum())}/{n_kept} sources at least once ({sum(stats['n_refused'])} "
          "refusals; a refused row keeps its previous state)")
    print(f"definitions: values clipped at epsilon={eps:g}; dex = |log10(yp) - log10(yt)|; "
          f"frac = |yp - yt| / (|yt| + {FRAC_ERR_FLOOR:g}) (legacy, for continuity)")
    print("share of scored truth below " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in stats["truth_share_below"].items())
          + f" (the fractional error's {FRAC_ERR_FLOOR:g} floor hides relative error below it)")

    names = [s.replace("_evolution", "") for s in species_keys]
    i90 = SPECIES_PCTLS.index(90)
    print(f"\n{'step':>5} " + " ".join(f"frac_p{p:<2}" for p in PCTLS) + "   dex_p90  persistence_dex_p90  n_finite  n_refused")
    for k in range(K):
        if (k + 1) in report:
            print(f"{k + 1:>5} " + " ".join(f"{v:7.1f}%" for v in stats["frac"][k])
                  + f"   {stats['dex_p90'][k]:6.3f}  {pstats['dex_p90'][k]:19.3f}  {stats['n_finite'][k]:8d}  {stats['n_refused'][k]:9d}")
    dt_w = dt_seconds[:, 0]
    dt_rule = (f"each window's own constant dt (log-uniform in [{pcfg.dt_min:g}, {pcfg.dt_max:g}] s; no fixed-dt set yet); "
               f"this selection: min {dt_w.min():.4g} s, median {np.median(dt_w):.4g} s, max {dt_w.max():.4g} s")
    # Per-stratum percentiles at the report steps (every headline step is a report step).
    def stratum_rows(mask: np.ndarray) -> Dict[str, Any] | None:
        if not mask.any():
            return None
        return {
            "n": int(mask.sum()),
            "headline_species_dex_p90": {
                str(s): {w: _pctl_fixed_pool(st["dex_at_report"][report.index(s)][mask], 90, axis=0).tolist()
                         for w, st in (("model", stats), ("persistence", pstats))}
                for s in headline
            },
            "pooled_dex_p90_at_report_steps": {
                w: [float(_pctl_fixed_pool(st["dex_at_report"][r][mask].ravel(), 90)) for r in range(len(report))]
                for w, st in (("model", stats), ("persistence", pstats))
            },
        }

    by_stratum = {k: stratum_rows(m) for k, m in strata.items()}
    print(f"\n[strata] {n_steady} of {n_kept} windows are steady state (truth within {STEADY_STATE_MAX_DEX:g} dex of "
          f"the start state at every step and species, so persistence is nearly exact); {n_kept - n_steady} evolving")
    print("\n=== per-species dex p90 at the headline steps by stratum (evolving: the recommended headline; "
          "random-dt windows) ===")
    print(f"step s = the s-th rollout step from start index {ROLLOUT_START_INDEX}; dt = {dt_rule}")
    def print_headline(label: str, n_w: int, rows_by_step: Dict[str, Any]) -> None:
        print(f"--- {label} ({n_w} windows) ---")
        for s in headline:
            row, prow = rows_by_step[str(s)]["model"], rows_by_step[str(s)]["persistence"]
            print(f"@ step {s}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, row, strict=True)))
            print(f"@ step {s} persistence: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, prow, strict=True)))

    for key, label in (("evolving", "evolving"), ("steady_state", "steady state")):
        if by_stratum[key] is None:
            print(f"--- {label} (0 windows) ---")
        else:
            print_headline(label, by_stratum[key]["n"], by_stratum[key]["headline_species_dex_p90"])
    print_headline("all windows", n_kept, {str(s): {"model": stats["species_dex"][s - 1][i90],
                                                     "persistence": pstats["species_dex"][s - 1][i90]} for s in headline})

    # Loss against persistence (ratio of means; skill = 1 - ratio), overall and by window-start decade,
    # over the evolving windows: on a steady-state window persistence is nearly exact, so a ratio there
    # is meaningless (gate 1's population since 2026-10-01). Steady-state windows are reported by the
    # model's absolute log10 error instead.
    m_wl, p_wl = stats["window_loss"], pstats["window_loss"]
    m_wd, p_wd = stats["window_abs_dex"], pstats["window_abs_dex"]
    evolving = strata["evolving"]

    def vs_persistence(mask: np.ndarray) -> Dict[str, Any]:
        m, p = float(m_wl[mask].mean()), float(p_wl[mask].mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = float(np.float64(m) / np.float64(p))
        return {"n": int(mask.sum()), "model_loss": m, "persistence_loss": p, "ratio": r, "skill": 1.0 - r}

    def abs_error(mask: np.ndarray) -> Dict[str, Any]:
        return {"n": int(mask.sum()), "model_mean_dex": float(m_wd[mask].mean()),
                "model_p90_dex": float(_pctl_fixed_pool(m_wd[mask], 90)),
                "persistence_mean_dex": float(p_wd[mask].mean())}

    decade = np.floor(np.log10(t_window)).astype(int)
    decades = sorted(set(decade.tolist()))
    # A decade without evolving windows has no ratio; it appears in the steady-state table only.
    by_decade = [{"t_lo_s": 10.0 ** d, "t_hi_s": 10.0 ** (d + 1), **vs_persistence(evolving & (decade == d)),
                  "n_steady_state": int((steady & (decade == d)).sum())}
                 for d in decades if (evolving & (decade == d)).any()]
    steady_by_decade = [{"t_lo_s": 10.0 ** d, "t_hi_s": 10.0 ** (d + 1), **abs_error(steady & (decade == d))}
                        for d in decades if (steady & (decade == d)).any()]
    overall = vs_persistence(evolving) if evolving.any() else None
    pooled = vs_persistence(np.ones_like(m_wl, dtype=bool))
    steady_overall = abs_error(steady) if steady.any() else None
    if by_stratum["evolving"] is not None:
        by_stratum["evolving"]["loss_vs_persistence"] = overall
    if by_stratum["steady_state"] is not None:
        by_stratum["steady_state"]["abs_log10_error"] = steady_overall
    loss_label = (f"the run's val_loss definition: {loss_def['k']} step(s) from start index {ROLLOUT_START_INDEX}, "
                  f"steps >= {loss_def['skip']} weighted equally, lambda_log10_mae={loss_def['lambda_log10_mae']:g}, "
                  f"lambda_z_mse={loss_def['lambda_z_mse']:g}; float64 from the export's outputs, clipped at "
                  f"epsilon (not training's val_loss); window start = t_start + {ROLLOUT_START_INDEX} x dt")
    abs_label = ("per window, mean |log10 yp - log10 yt| over species and the val_loss steps (same step weights; "
                 "the loss's log10-MAE term), clipped at epsilon; +inf if diverged")
    print(f"\n=== loss vs persistence, evolving windows ({loss_label}) ===")
    print("(steady: the decade's steady-state windows, left out of the ratio; a decade with none evolving is not listed)")
    print(f"{'window start (s)':<20} {'n':>6} {'steady':>6} {'model_loss':>12} {'persistence':>12} {'ratio':>8} {'skill':>8}")
    for row in by_decade:
        span = f"[{row['t_lo_s']:.0e}, {row['t_hi_s']:.0e})"
        print(f"{span:<20} {row['n']:>6} {row['n_steady_state']:>6} {row['model_loss']:12.4e} "
              f"{row['persistence_loss']:12.4e} {row['ratio']:8.4f} {row['skill']:8.4f}")
    for label, row, n_st in (("evolving", overall, n_steady), ("all, pooled (diag.)", pooled, n_steady)):
        if row is None:
            print(f"{label:<20} {0:>6} {n_st:>6} (no evolving windows)")
        else:
            print(f"{label:<20} {row['n']:>6} {n_st:>6} {row['model_loss']:12.4e} {row['persistence_loss']:12.4e} "
                  f"{row['ratio']:8.4f} {row['skill']:8.4f}")
    print("all, pooled: every window, steady state included (training's val_skill population); a diagnostic, "
          "not gate 1")
    print(f"\n=== steady-state windows: absolute log10 error in dex ({abs_label}) ===")
    if steady_overall is None:
        print("(no steady-state windows)")
    else:
        print(f"{'window start (s)':<20} {'n':>6} {'model_mean':>11} {'model_p90':>11} {'persistence_mean':>17}")
        for row in steady_by_decade + [dict(steady_overall, t_lo_s=None)]:
            span = "all steady state" if row["t_lo_s"] is None else f"[{row['t_lo_s']:.0e}, {row['t_hi_s']:.0e})"
            print(f"{span:<20} {row['n']:>6} {row['model_mean_dex']:11.4e} {row['model_p90_dex']:11.4e} "
                  f"{row['persistence_mean_dex']:17.4e}")
    late = [r for r in by_decade if r["t_lo_s"] >= GATE1_LATE_START_S]
    gate1 = {
        "provisional": True, "enforced": False,
        "population": "evolving windows only (steady-state windows are reported apart by absolute log10 error)",
        "applies": not bool(cfg["training"]["autoregressive_training"]["enabled"]),
        "judged": overall is not None,
        "max_overall_ratio": GATE1_MAX_LOSS_RATIO, "max_decade_ratio": GATE1_MAX_DECADE_RATIO,
        "late_start_s": GATE1_LATE_START_S,
        "overall_pass": overall is not None and overall["ratio"] <= GATE1_MAX_LOSS_RATIO,
        "decades_pass": all(r["ratio"] < GATE1_MAX_DECADE_RATIO for r in by_decade),
        "late_decades_pass": all(r["ratio"] < GATE1_MAX_DECADE_RATIO for r in late),
        "failing_decades_s": [r["t_lo_s"] for r in by_decade if not r["ratio"] < GATE1_MAX_DECADE_RATIO],
    }
    gate1["pass"] = gate1["judged"] and gate1["overall_pass"] and gate1["decades_pass"]
    pf = {True: "PASS", False: "FAIL"}
    if not gate1["applies"]:
        print("gate 1 is the stage-1 (one-jump) gate; for this autoregressive run the tables are diagnostics")
    elif not gate1["judged"]:
        print(f"gate 1 (PROVISIONAL, reported only): no evolving windows in this selection, so it is not judged. "
              f"GATE 1: {pf[False]}")
    else:
        print(f"\ngate 1 (PROVISIONAL, reported only; evolving windows only): overall ratio {overall['ratio']:.4f} "
              f"<= {GATE1_MAX_LOSS_RATIO:g}: {pf[gate1['overall_pass']]}")
        print(f"gate 1: ratio < {GATE1_MAX_DECADE_RATIO:g} in every window-start decade with evolving windows "
              f"({len(by_decade)} decades): {pf[gate1['decades_pass']]}"
              + (f" (failing decades from {gate1['failing_decades_s']} s)" if gate1["failing_decades_s"] else ""))
        print(f"gate 1: ratio < {GATE1_MAX_DECADE_RATIO:g} in the decades from {GATE1_LATE_START_S:g} s on "
              f"({len(late)} decades): {pf[gate1['late_decades_pass']]}")
        print(f"GATE 1: {pf[gate1['pass']]}")
        print(f"gate 1 leaves out the {n_steady} steady-state windows (no threshold; their absolute error is above)")

    split = EVAL_SPLIT
    curve = run_dir / f"rollout_error_curve_{split}.csv"
    with open(curve, "w") as f:
        f.write("step," + ",".join(f"frac_p{p}" for p in PCTLS) + ",dex_p90,n_finite,n_refused,"
                + ",".join(f"dex_p90_{n}" for n in names) + ",persistence_dex_p90,"
                + ",".join(f"persistence_dex_p90_{n}" for n in names) + "\n")
        for k in range(K):
            sp90 = stats["species_dex"][k][i90]
            pp90 = pstats["species_dex"][k][i90]
            f.write(f"{k + 1}," + ",".join(f"{v:.6f}" for v in stats["frac"][k])
                    + f",{stats['dex_p90'][k]:.6f},{stats['n_finite'][k]},{stats['n_refused'][k]},"
                    + ",".join(f"{v:.6f}" for v in sp90)
                    + f",{pstats['dex_p90'][k]:.6f}," + ",".join(f"{v:.6f}" for v in pp90) + "\n")
    long = run_dir / f"rollout_error_species_{split}.csv"
    with open(long, "w") as f:
        f.write("step,species,n_finite," + ",".join(f"dex_p{p}" for p in SPECIES_PCTLS) + ",bias_median_dex,bias_mean_dex\n")
        for k in range(K):
            for j, n in enumerate(names):
                f.write(f"{k + 1},{n},{stats['n_finite'][k]},"
                        + ",".join(f"{stats['species_dex'][k][i][j]:.6f}" for i in range(len(SPECIES_PCTLS)))
                        + f",{stats['bias_median'][k][j]:.6f},{stats['bias_mean'][k][j]:.6f}\n")
    dex_at_report = np.stack(stats["dex_at_report"])  # [R, N, S]
    per_source = run_dir / f"rollout_error_{split}_per_source.npz"
    np.savez(per_source, dex_at_report=dex_at_report, report_steps=np.asarray(report),
             rows=np.asarray(kept_rows), selection_sha256=np.asarray(selection_sha),
             window_start_s=t_window, window_loss=m_wl, persistence_window_loss=p_wl, steady_state=steady,
             window_abs_log10_error=m_wd, persistence_window_abs_log10_error=p_wd)
    shards_by_file = []
    cum = 0
    for r in summary["raw_files"]:
        n_sh = int(r["shards_by_split"][split])
        shards_by_file.append((r["name"], cum, cum + n_sh))
        cum += n_sh
    file_of = np.array([next(name for name, lo, hi in shards_by_file if lo <= s < hi) for s, _ in kept_rows])
    per_file = {
        name: {"sources": int((file_of == name).sum()),
               "pooled_dex_p90_at_report_steps": [float(_pctl_fixed_pool(dex_at_report[r, file_of == name].ravel(), 90))
                                                 for r in range(len(report))] if (file_of == name).any() else None}
        for name, _, _ in shards_by_file
    }
    provenance = {
        "label": DATA_LABEL,
        "run_dir": str(run_dir),
        "export_path": str(export_path),
        "export_sha256": file_sha256(export_path),
        "export_metadata": {k: meta.get(k) for k in ("checkpoint_path", "checkpoint_sha256", "config_path", "export_dtype",
                                                    "epsilon", "units", "oor_policy", "oor_box", "norm_sha256",
                                                    "weights_source", "epoch", "checkpoint_score")},
        "normalization_path": str(norm_path),
        "normalization_sha256": file_sha256(norm_path),
        "split": split,
        "shard_root": str(shard_root),
        "shards": [p.name for p in shards],
        "rows_total": n_rows_total,
        "sources_total": n_sources_total,
        "sources_selected": N,
        "sources_out_of_box_dropped": int(oor.sum()),
        "sources_per_raw_file": per_file,
        "selection": {"seed": EVAL_SEED, "n_sources": EVAL_N_SOURCES, "rule": "uniform sources by (P, T) key, one uniform window each",
                      "sha256_of_shard_row_list": selection_sha},
        "T": T, "K": K, "start_index": ROLLOUT_START_INDEX, "report_steps": report, "headline_steps": headline,
        "headline_dt": dt_rule,
        "loss_vs_persistence": {"definition": loss_label, "population": "evolving windows (gate 1's)",
                                "overall": overall, "by_window_start_decade": by_decade,
                                "all_windows_pooled": pooled},
        "steady_state_abs_log10_error": {"definition": abs_label, "overall": steady_overall,
                                         "by_window_start_decade": steady_by_decade},
        "strata": {
            "definition": ("steady_state: the window's truth stays within max_dex of its start state, "
                           "max over the K steps and species of |log10 y_true - log10 y_start| (clipped at "
                           "epsilon) <= max_dex, so persistence is nearly exact; evolving: the other windows "
                           "(the recommended headline and gate 1's population); the curve CSVs pool all "
                           "windows; steady-state windows are scored by absolute log10 error, not a ratio"),
            "max_dex": STEADY_STATE_MAX_DEX,
            **by_stratum,
        },
        "gate1": gate1,
        "clip_eps": eps, "frac_err_floor": FRAC_ERR_FLOOR,
        "truth_share_below": stats["truth_share_below"],
        "infer_dtype": str(infer_dtype), "host_dtype": str(P.HOST_DTYPE), "host_update": "testing/export.py host_step",
        "host_step_refusals": {"total": sum(stats["n_refused"]), "sources": int(stats["sources_refused"].sum())},
        "torch": torch.__version__, "numpy": np.__version__,
        "preprocessing": {k: summary.get(k) for k in ("windows_written_by_split", "groups_kept", "group_rejects", "chunk_rejects")},
        "bootstrap": {"n": EVAL_BOOTSTRAP_N, "seed": EVAL_BOOTSTRAP_SEED, "level": EVAL_CI_LEVEL,
                      "species_pctls": list(SPECIES_PCTLS), "species": names, **_bootstrap_ci(dex_at_report)},
        "paired_with_source_run": _paired_difference(run_dir, dex_at_report, selection_sha, per_source.name),
    }
    side = run_dir / f"rollout_error_{split}.json"
    side.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"\nwrote {curve}\nwrote {long}\nwrote {side}\nwrote {per_source}")


if __name__ == "__main__":
    main()
