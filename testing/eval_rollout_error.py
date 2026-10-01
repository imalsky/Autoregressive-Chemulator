#!/usr/bin/env python3
"""Rollout-error curves of an exported 1-step model over a seeded sample of one split (diagnostic).

Rolls the run's exported CPU model (export_cpu_dynB_1step_phys.pt2 in the run directory named by
AUTOCHEM_CONFIG_PATH, testing/run_dir.py) autoregressively at each window's own fixed dt over a
reproducible sample of EVAL_SPLIT: EVAL_N_SOURCES source trajectories drawn with EVAL_SEED, one
seeded window per source. The windows are the processed random-dt shards (dt log-uniform in the
trained range), so K <= T - 1 - ROLLOUT_START_INDEX (98 today) and every row is labelled
"random-dt, K <= 98": no fixed-dt evaluation set exists yet.

Reported per rollout step (no pass/fail thresholds; spec 6.3):
- per-species |log10(yp) - log10(yt)| p50 / p90 in dex (recommended headline; the author confirms
  the paper target) and the median / mean signed log10(yp / yt) (bias); 10% fractional error is
  about 0.041 dex for an overshoot and 0.046 dex for an undershoot;
- the legacy pooled fractional-error percentiles |yp - yt| / (|yt| + FRAC_ERR_FLOOR) for continuity
  (the floor hides trace-species error; the share of scored truth below 1e-12 and 1e-20 is printed);
- n_finite. Percentiles keep a fixed population: a diverged (non-finite) entry ranks as +inf error,
  so an inf percentile means more than that tail fraction diverged.
Bootstrap CIs over sources are given at the report steps. Sources whose dt, P or T lies outside the
export's trained box (NaN rows by the export contract) are dropped before the rollout and counted
apart from divergence. If this run was started with weights_only from another run whose evaluation
on the same selection exists, a paired bootstrap of the difference is added.

Run (once per stage, e.g. after export.py):
  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/eval_rollout_error.py
Outputs (in the run directory, named by split): rollout_error_curve_<split>.csv,
rollout_error_species_<split>.csv, rollout_error_<split>.json (provenance and CIs) and
rollout_error_<split>_per_source.npz (per-source errors at the report steps, for paired comparisons).
The CPU export runs at full float32 (TF32 does not apply on CPU).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))

import export as E  # noqa: E402  (export's baked normalizer: the round-trip check; src/utils fingerprint)
import predictions as P  # noqa: E402  (its independent numpy inverse normalization and loaders)
from run_dir import resolve_run_dir  # noqa: E402
from utils import file_sha256, normalization_sha256  # noqa: E402

# Every gate, flip and trigger decision uses validation; 'test' only for the single final paper run.
EVAL_SPLIT = "validation"
# None => the run config's paths.processed_dir; or the root of a designed evaluation set in the
# same shard schema and normalization (checked by norm_sha256).
EVAL_SHARD_ROOT: str | None = None
EVAL_N_SOURCES = 4000          # inherited (the former trajectory cap); one window per source
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


@torch.inference_mode()
def _rollout_stats(model, y0_phys, g_phys, dt_seconds, y_true, *, eps: float, b_max: int, report_idx: List[int]):
    """Steps in the outer loop, row chunks of at most b_max in the inner loop; keep only the [N, S] state.

    Returns per-step pooled frac percentiles, pooled dex p90, per-species dex percentiles and bias,
    n_finite, the truth-below-level shares, and the per-source dex at the report steps.
    """
    N, K, S = y_true.shape
    state = torch.from_numpy(y0_phys.astype(np.float32))
    g = torch.from_numpy(g_phys.astype(np.float32))
    dt = torch.from_numpy(dt_seconds.astype(np.float32))
    out: Dict[str, Any] = {"frac": [], "dex_p90": [], "species_dex": [], "bias_median": [], "bias_mean": [],
                           "n_finite": [], "dex_at_report": []}
    below = {lvl: 0 for lvl in TRUTH_DISCLOSURE_LEVELS}
    for k in range(K):
        nxt = [model(state[i:i + b_max], dt[i:i + b_max, k], g[i:i + b_max]) for i in range(0, N, b_max)]
        state = torch.cat(nxt, dim=0)
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
    out["truth_share_below"] = {f"{lvl:g}": below[lvl] / float(N * K * S) for lvl in below}
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
    root_manifest = P._load_json(shard_root / "normalization.json")
    if normalization_sha256(root_manifest) != meta["norm_sha256"]:
        raise ValueError(f"{shard_root} is not normalized like {export_path.name} (norm_sha256 differs)")
    summary = P._load_json(shard_root / "preprocessing_summary.json")
    species_keys = list(manifest["species_variables"])
    global_keys = list(manifest["global_variables"])
    eps = float(manifest["epsilon"])
    b_max = int(meta["dynamic_batch"]["max"])
    print(f"[model] {export_path}  (CPU float32; TF32 does not apply)")
    print(f"[data]  {shard_root}/{EVAL_SPLIT}  ({DATA_LABEL})")

    shards = sorted((shard_root / EVAL_SPLIT).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shard_root / EVAL_SPLIT}")
    rows, n_sources_total, n_rows_total = _select_rows(
        shards, max_rows_per_source=int(summary["config"]["samples_per_source_trajectory"]))
    selection_sha = hashlib.sha256(json.dumps([[shards[s].name, r] for s, r in rows]).encode()).hexdigest()
    y_mat, g_mat, dt_norm_mat = _load_rows(shards, rows)
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

    report = sorted({min(s, K) for s in REPORT_STEPS})
    headline = sorted({min(s, K) for s in HEADLINE_STEPS})
    stats = _rollout_stats(model, y0_phys, g_phys, dt_seconds, y_true, eps=eps, b_max=b_max,
                           report_idx=[s - 1 for s in report])
    n_kept = int(keep.sum())
    print(f"non-finite (diverged) sources at the last step: {n_kept - stats['n_finite'][-1]}/{n_kept}")
    print(f"definitions: values clipped at epsilon={eps:g}; dex = |log10(yp) - log10(yt)|; "
          f"frac = |yp - yt| / (|yt| + {FRAC_ERR_FLOOR:g}) (legacy, for continuity)")
    print("share of scored truth below " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in stats["truth_share_below"].items())
          + f" (the fractional error's {FRAC_ERR_FLOOR:g} floor hides relative error below it)")

    names = [s.replace("_evolution", "") for s in species_keys]
    print(f"\n{'step':>5} " + " ".join(f"frac_p{p:<2}" for p in PCTLS) + "   dex_p90  n_finite")
    for k in range(K):
        if (k + 1) in report:
            print(f"{k + 1:>5} " + " ".join(f"{v:7.1f}%" for v in stats["frac"][k])
                  + f"   {stats['dex_p90'][k]:6.3f}  {stats['n_finite'][k]}")
    print("\n=== per-species dex p90 at the headline steps (recommended headline; random-dt windows) ===")
    for s in headline:
        row = stats["species_dex"][s - 1][SPECIES_PCTLS.index(90)]
        print(f"@ step {s}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, row, strict=True)))

    split = EVAL_SPLIT
    curve = run_dir / f"rollout_error_curve_{split}.csv"
    with open(curve, "w") as f:
        f.write("step," + ",".join(f"frac_p{p}" for p in PCTLS) + ",dex_p90,n_finite,"
                + ",".join(f"dex_p90_{n}" for n in names) + "\n")
        for k in range(K):
            sp90 = stats["species_dex"][k][SPECIES_PCTLS.index(90)]
            f.write(f"{k + 1}," + ",".join(f"{v:.6f}" for v in stats["frac"][k])
                    + f",{stats['dex_p90'][k]:.6f},{stats['n_finite'][k]}," + ",".join(f"{v:.6f}" for v in sp90) + "\n")
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
             rows=np.asarray(kept_rows), selection_sha256=np.asarray(selection_sha))
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
        "clip_eps": eps, "frac_err_floor": FRAC_ERR_FLOOR,
        "truth_share_below": stats["truth_share_below"],
        "infer_dtype": str(infer_dtype), "torch": torch.__version__, "numpy": np.__version__,
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
