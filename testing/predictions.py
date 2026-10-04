#!/usr/bin/env python3
"""
Autoregressive rollout evaluation for exported 1-step physical-space models.

Export signature:
    y_next_phys = model(y_phys, dt_seconds, g_phys)

Run (the run directory is paths.work_dir of the config, testing/run_dir.py):
    AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/predictions.py

The CPU export runs at full float32 (TF32 does not apply on CPU). This script imports nothing from
src/ (standalone), so it keeps its own normalization helpers.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]

from run_dir import resolve_run_dir  # noqa: E402  (testing/ is this script's directory)

# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

# Model/artifact selection: the run comes from AUTOCHEM_CONFIG_PATH (testing/run_dir.py).
MODEL_EXPORT_FILE: str = "export_cpu_dynB_1step_phys.pt2"

# Inference settings.
INFER_DEVICE: torch.device = torch.device("cpu")
INFER_DTYPE: torch.dtype | None = None  # None => use export metadata dtype exactly

# Data selection. Validation: this script runs after every training (diagnostics and decisions use
# the validation split); the test split is for the single final paper evaluation.
EVAL_SPLIT: str = "validation"
# Fixed default window (the first sample of the first shard of the split), so every run, and the
# stage-1 and stage-2 figures (both stages read the same processed_dir), show the same trajectory.
# The pick is stable only while the processed dataset is unchanged: shards carry no trajectory id, so
# a rebuild with other raw files or settings moves it. Set an index to None to opt in to a seeded
# random pick instead (EVAL_*_RANDOM_SEED).
EVAL_SHARD_INDEX: int | None = 0
EVAL_SHARD_RANDOM_SEED: int = 0
EVAL_SAMPLE_INDEX: int | None = 0
EVAL_SAMPLE_RANDOM_SEED: int = 0

# Rollout window. Inherited: index 0 of every window is skipped with no recorded rationale;
# changing it changes the reported errors (testing/eval_rollout_error.py uses the same value).
ROLLOUT_START_INDEX: int = 1
ROLLOUT_STEPS: int = 50

# Plotting/output.
PLOTS_SUBDIR: str = "plots"  # Relative to MODEL_RUN_DIR
PLOT_TIME_LOG_SCALE: bool = False  # If True, use log-scale on time axis

# Figure style: the user's scientific-plotting rules, inlined (repo scripts do not import them).
# Okabe-Ito cycle (yellow dropped), charcoal ink, serif text, ticks in on all sides, frameless
# 10 pt legend, zero x-margin and the style's 0.1 y-margin (limits from the data), 1.5 pt lines, PNG at 300 dpi.
_CYC = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
_INK = "#2b2b2b"
_RC = {
    "font.size": 12, "font.family": "serif",
    "font.serif": ["Times New Roman", "Palatino", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.5, "axes.labelsize": "large", "axes.labelpad": 5, "axes.xmargin": 0, "axes.ymargin": 0.1,
    "axes.formatter.limits": (-6, 6), "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": False, "axes.prop_cycle": cycler(color=list(_CYC)),
    "xtick.direction": "in", "xtick.top": True, "xtick.minor.visible": True, "xtick.labelsize": "large",
    "xtick.major.size": 8, "xtick.major.width": 1.3, "xtick.minor.size": 4, "xtick.minor.width": 1,
    "xtick.major.pad": 6,
    "ytick.direction": "in", "ytick.right": True, "ytick.minor.visible": True, "ytick.labelsize": "large",
    "ytick.major.size": 8, "ytick.major.width": 1.3, "ytick.minor.size": 4, "ytick.minor.width": 1,
    "ytick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 10, "legend.borderaxespad": 1,
    "legend.handlelength": 1.5, "legend.labelspacing": 0.3,
    "lines.linewidth": 1.5, "savefig.bbox": "tight",
}
_DPI = 300
# Square panels of len(_CYC) species each (one cycle colour per species within a panel); allowed
# layouts only, so at most 4 * 6 = 24 species.
_PANEL_LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}
_PANEL_SIZE_IN = (5.33, 5.2)  # per panel (width, height)
# Overlay recipe (several coincident pairs in a panel): the truth as a tint of the species colour
# underneath, the prediction in the full colour dashed on top, at every step. One style source for
# the plotted lines and the legend proxies.
_TINT_KEEP = 0.6
_TRUTH_KW = {"lw": 2.5, "ls": "-", "solid_capstyle": "round"}
_PRED_KW = {"lw": 1.2, "ls": "--"}
_LEGEND_CORNERS = ("upper right", "upper left", "lower right", "lower left")
# Points sampled along each drawn segment when scoring a legend corner.
_SEGMENT_SAMPLES = 20

# Fixed global set, in this order. This standalone script
# imports nothing from src/, so it keeps its own copy of src/model.py:_REQUIRED_GLOBALS.
REQUIRED_GLOBALS: Tuple[str, str] = ("P", "T")

# Denominator floor of the legacy fractional error |yp - yt| / (|yt| + FRAC_ERR_FLOOR) (inherited
# from the sibling one-step paper's definition): it hides relative error on trace species below
# about 1e-12 (24% of scored truth on the 4-file data), so per-species dex error is the headline;
# the fractional numbers are kept for continuity only.
FRAC_ERR_FLOOR: float = 1e-12


def _resolve_export_path(run_dir: Path) -> Path:
    return (run_dir / MODEL_EXPORT_FILE).resolve()


def _resolve_output_dir(run_dir: Path) -> Path:
    return (run_dir / PLOTS_SUBDIR).resolve()


def _parse_export_dtype(raw: Any) -> torch.dtype:
    if not isinstance(raw, str) or not raw.strip():
        raise KeyError("export metadata missing export_dtype")

    value = raw.strip().lower()
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported export_dtype in metadata: {raw!r}")


def _select_infer_dtype(meta: Mapping[str, Any]) -> torch.dtype:
    export_dtype = _parse_export_dtype(meta.get("export_dtype"))
    if INFER_DTYPE is not None and INFER_DTYPE != export_dtype:
        raise ValueError(
            f"INFER_DTYPE={str(INFER_DTYPE).replace('torch.', '')} does not match "
            f"export metadata dtype={str(export_dtype).replace('torch.', '')}"
        )
    return export_dtype if INFER_DTYPE is None else INFER_DTYPE


def _load_export_module(export_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any], torch.dtype]:
    extra_files = {"metadata.json": ""}
    ep = torch.export.load(str(export_path), extra_files=extra_files)
    meta_raw = extra_files.get("metadata.json", "")
    if not meta_raw:
        raise RuntimeError("export is missing embedded metadata.json")
    meta = json.loads(meta_raw)
    if not isinstance(meta, dict):
        raise TypeError("embedded metadata.json must be a JSON object")
    infer_dtype = _select_infer_dtype(meta)
    model = ep.module().to(device=INFER_DEVICE, dtype=infer_dtype)
    return model, meta, infer_dtype


def _resolve_normalization_path(meta: Mapping[str, Any]) -> Path:
    raw = meta.get("normalization_path")
    if not isinstance(raw, str) or not raw.strip():
        raise KeyError("export metadata missing normalization_path")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"normalization_path from export metadata not found: {p}")
    return p


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return obj


def _pick_methods_map(manifest: Mapping[str, Any]) -> Mapping[str, str]:
    methods = manifest.get("normalization_methods")
    if not isinstance(methods, Mapping):
        raise KeyError("normalization.json missing normalization_methods mapping")
    return methods  # type: ignore[return-value]


def _canonical_method(method: str) -> str:
    m = str(method).lower().strip()
    return m


def _dt_norm_to_seconds(dt_norm: np.ndarray, manifest: Mapping[str, Any]) -> np.ndarray:
    dt = manifest.get("dt")
    if not isinstance(dt, Mapping):
        raise KeyError("normalization.json missing dt")
    if "log_min" not in dt or "log_max" not in dt:
        raise KeyError("normalization.json missing dt.log_min/dt.log_max")

    a = float(dt["log_min"])
    b = float(dt["log_max"])
    return np.power(10.0, dt_norm.astype(np.float64) * (b - a) + a, dtype=np.float64)


def _denormalize_species(y_z: np.ndarray, species_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")
    methods = _pick_methods_map(manifest)

    for k in species_keys:
        m = _canonical_method(str(methods.get(k, "")))
        if m != "log-standard":
            raise ValueError(f"species normalization must be log-standard, got {m!r} for {k!r}")

    mu = np.array([float(stats[k]["log_mean"]) for k in species_keys], dtype=np.float64)
    sd = np.array([float(stats[k]["log_std"]) for k in species_keys], dtype=np.float64)
    return np.power(10.0, y_z.astype(np.float64) * sd + mu, dtype=np.float64)


def _denormalize_globals(g_z: np.ndarray, global_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    if list(global_keys) != list(REQUIRED_GLOBALS):
        raise ValueError(f"global_variables must be exactly {list(REQUIRED_GLOBALS)}")

    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")
    methods = _pick_methods_map(manifest)

    out = np.empty((len(global_keys),), dtype=np.float64)
    for i, key in enumerate(global_keys):
        st = stats[key]
        z = float(g_z[i])
        m = _canonical_method(str(methods.get(key, "")))

        if m == "min-max":
            lo = float(st["min"])
            hi = float(st["max"])
            out[i] = z * (hi - lo) + lo
            continue

        if m == "log-min-max":
            lo = float(st["log_min"])
            hi = float(st["log_max"])
            out[i] = 10.0 ** (z * (hi - lo) + lo)
            continue

        if m == "standard":
            mu = float(st["mean"])
            sd = float(st["std"])
            out[i] = z * sd + mu
            continue

        if m == "log-standard":
            mu = float(st["log_mean"])
            sd = float(st["log_std"])
            out[i] = 10.0 ** (z * sd + mu)
            continue

        raise ValueError(f"unsupported global normalization method {m!r} for {key!r}")

    return out


def _choose_shard(processed_dir: Path, *, split: str, shard_index: int | None) -> Tuple[int, Path]:
    shard_dir = processed_dir / split
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards found under {shard_dir}")

    if shard_index is None:
        rng = np.random.default_rng(EVAL_SHARD_RANDOM_SEED)
        picked = int(rng.integers(0, len(shards)))
        print(f"[data] randomly selected shard index {picked} of {len(shards)}, seed={EVAL_SHARD_RANDOM_SEED}")
        return picked, shards[picked]

    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"SHARD_INDEX out of range: {shard_index} (found {len(shards)} shards)")

    return shard_index, shards[shard_index]


def _load_shard(shard_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(shard_path) as f:
        y_mat = f["y_mat"].astype(np.float32, copy=False)
        g_mat = f["globals"].astype(np.float32, copy=False)
        dt_norm_mat = f["dt_norm_mat"].astype(np.float32, copy=False)
    return y_mat, g_mat, dt_norm_mat


def _choose_sample(n_samples: int, *, sample_index: int | None) -> int:
    if n_samples <= 0:
        raise ValueError(f"selected shard has no samples/chunks (N={n_samples})")

    if sample_index is None:
        rng = np.random.default_rng(EVAL_SAMPLE_RANDOM_SEED)
        picked = int(rng.integers(0, n_samples))
        print(f"[data] randomly selected sample index {picked} of {n_samples}, seed={EVAL_SAMPLE_RANDOM_SEED}")
        return picked

    if sample_index < 0 or sample_index >= n_samples:
        raise IndexError(f"SAMPLE_INDEX out of range: {sample_index} (shard has N={n_samples})")

    return sample_index


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    *,
    y0_phys: np.ndarray,
    g_phys: np.ndarray,
    dt_seconds: np.ndarray,
    infer_dtype: torch.dtype,
) -> np.ndarray:
    if g_phys.shape != (len(REQUIRED_GLOBALS),):
        raise ValueError(f"g_phys must have shape ({len(REQUIRED_GLOBALS)},), got {g_phys.shape}")

    y = torch.from_numpy(y0_phys.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=infer_dtype).unsqueeze(0)
    g = torch.from_numpy(g_phys.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=infer_dtype).unsqueeze(0)
    dt_all = torch.from_numpy(dt_seconds.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=infer_dtype)

    preds: List[torch.Tensor] = []
    for k in range(int(dt_all.shape[0])):
        y = model(y, dt_all[k : k + 1], g)
        preds.append(y[0].detach().cpu())
    return torch.stack(preds, dim=0).numpy().astype(np.float64, copy=False)


def _tint(color: str) -> Tuple[float, float, float]:
    """`color` blended toward white, keeping _TINT_KEEP of it (the truth underlay)."""
    return tuple(1.0 - _TINT_KEEP + _TINT_KEEP * v for v in to_rgb(color))


def _drawn_points(ax, line) -> np.ndarray:
    """Display coordinates of a line's points, plus _SEGMENT_SAMPLES points along each drawn segment
    (so a line crossing a legend counts, not only its vertices)."""
    xy = ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if line.get_linestyle() in ("None", "", " ") or len(xy) < 2:
        return xy
    t = np.linspace(0.0, 1.0, _SEGMENT_SAMPLES)[:, None, None]
    return (xy[:-1] * (1.0 - t) + xy[1:] * t).reshape(-1, 2)


def _legend(ax, handles: list) -> None:
    """Legend in the corner covering the fewest plotted points, one column then two; matplotlib's
    'best' when every corner covers data. Never changes the limits."""
    fig = ax.figure
    for ncol in (1, 2):
        scores = {}
        for loc in _LEGEND_CORNERS:
            leg = ax.legend(handles=handles, loc=loc, ncol=ncol)
            fig.canvas.draw()
            x0, y0, x1, y1 = leg.get_window_extent().extents
            n = 0
            for line in ax.lines:
                xy = _drawn_points(ax, line)
                n += int(np.sum((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)))
            scores[loc] = n
        loc = min(scores, key=scores.get)
        if scores[loc] == 0:
            ax.legend(handles=handles, loc=loc, ncol=ncol)
            return
    ax.legend(handles=handles, loc="best")


def _species_label(key: str) -> str:
    """'CH4_evolution' -> 'CH$_{4}$' (digits as mathtext subscripts)."""
    return re.sub(r"(\d+)", r"$_{\1}$", key.replace("_evolution", ""))


def _plot(
    *,
    t_sec: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_keys: List[str],
    source: str,
    out_path: Path,
) -> None:
    """Truth (tinted) and prediction (dashed) per species, len(_CYC) species per square panel, by abundance."""
    n_species = len(species_keys)
    n_panels = math.ceil(n_species / len(_CYC))
    if n_panels not in _PANEL_LAYOUTS:
        raise ValueError(
            f"{n_species} species need {n_panels} panels of {len(_CYC)}; the allowed layouts hold at most "
            f"{max(_PANEL_LAYOUTS) * len(_CYC)} species"
        )
    rows, cols = _PANEL_LAYOUTS[n_panels]
    order = np.argsort(y_true.max(axis=0))[::-1]  # most abundant first

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            rows, cols, figsize=(_PANEL_SIZE_IN[0] * cols, _PANEL_SIZE_IN[1] * rows),
            squeeze=False, constrained_layout=True,
        )
        for p, ax in enumerate(axes.flat):
            idx = order[p * len(_CYC):(p + 1) * len(_CYC)]
            if idx.size == 0:
                ax.set_axis_off()
                continue
            handles = []
            for j, color in zip(idx, _CYC, strict=False):  # the last panel may hold fewer species
                ax.plot(t_sec, y_true[:, j], color=_tint(color), **_TRUTH_KW)
                ax.plot(t_sec, y_pred[:, j], color=color, **_PRED_KW)
                handles.append(Line2D([0], [0], color=color, label=_species_label(species_keys[j])))
            if p == 0:  # the line styles and the window's provenance, once
                handles += [
                    Line2D([0], [0], color=_tint(_INK), label=f"truth\n{source}", **_TRUTH_KW),
                    Line2D([0], [0], color=_INK, label="prediction", **_PRED_KW),
                ]
            if PLOT_TIME_LOG_SCALE:
                if float(np.min(t_sec)) <= 0.0:
                    raise ValueError("PLOT_TIME_LOG_SCALE=True requires strictly positive evaluation times.")
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("time since rollout start (s)")
            ax.set_ylabel("mixing ratio")
            ax.set_box_aspect(1)
            _legend(ax, handles)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)


def _print_errors(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float) -> None:
    """Print the rollout's errors; both arrays are clipped at the manifest epsilon (the species floor)."""
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)

    rel_err = np.abs(yp - yt) / (np.abs(yt) + FRAC_ERR_FLOOR)
    mae = np.mean(np.abs(yp - yt))
    log_err = np.mean(np.abs(np.log10(yp) - np.log10(yt)))

    print(f"Definitions: values clipped at epsilon={eps:g}; rel_err = |yp - yt| / (|yt| + {FRAC_ERR_FLOOR:g})")
    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


def main() -> None:
    run_dir = resolve_run_dir()
    export_path = _resolve_export_path(run_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"missing export: {export_path}")

    output_dir = _resolve_output_dir(run_dir)
    configured_dtype = "from_metadata" if INFER_DTYPE is None else str(INFER_DTYPE).replace("torch.", "")
    print(
        f"[config] export={export_path} run_dir={run_dir} "
        f"device={INFER_DEVICE.type} dtype={configured_dtype} split={EVAL_SPLIT}"
    )

    model, meta, infer_dtype = _load_export_module(export_path)
    print(
        f"[model] file={export_path.name} meta.export_device_tag={meta.get('export_device_tag')} "
        f"infer_dtype={str(infer_dtype).replace('torch.', '')}"
    )

    norm_path = _resolve_normalization_path(meta)
    processed_dir = norm_path.parent
    manifest = _load_json(norm_path)

    species_keys = list(manifest.get("species_variables") or [])
    if not species_keys:
        raise KeyError("normalization.json missing species_variables")
    global_keys = list(manifest.get("global_variables") or [])
    if global_keys != list(REQUIRED_GLOBALS):
        raise ValueError(f"normalization.json global_variables must be exactly {list(REQUIRED_GLOBALS)}")

    shard_index, shard_path = _choose_shard(processed_dir, split=EVAL_SPLIT, shard_index=EVAL_SHARD_INDEX)
    y_mat, g_mat, dt_norm_mat = _load_shard(shard_path)
    n_samples = int(y_mat.shape[0])
    sample_index = _choose_sample(n_samples, sample_index=EVAL_SAMPLE_INDEX)
    print(f"[data] split={EVAL_SPLIT} shard={shard_path.name} sample={sample_index} n_samples={n_samples}")

    y_z = y_mat[sample_index]
    g_z = g_mat[sample_index]
    dt_norm = dt_norm_mat[sample_index]

    T = int(y_z.shape[0])
    if ROLLOUT_START_INDEX < 0 or ROLLOUT_START_INDEX >= T - 1:
        raise IndexError(f"ROLLOUT_START_INDEX out of range: {ROLLOUT_START_INDEX} (trajectory has T={T})")

    max_steps = int((T - 1) - ROLLOUT_START_INDEX)
    if ROLLOUT_STEPS <= 0 or ROLLOUT_STEPS > max_steps:
        raise ValueError(f"ROLLOUT_STEPS must be in [1, {max_steps}], got {ROLLOUT_STEPS}")

    y0_z = y_z[ROLLOUT_START_INDEX]
    y_true_z = y_z[ROLLOUT_START_INDEX + 1 : ROLLOUT_START_INDEX + 1 + ROLLOUT_STEPS]
    dt_seq_norm = dt_norm[ROLLOUT_START_INDEX : ROLLOUT_START_INDEX + ROLLOUT_STEPS]

    y0_phys = _denormalize_species(y0_z[None, :], species_keys, manifest)[0]
    y_true_phys = _denormalize_species(y_true_z, species_keys, manifest)
    g_phys = _denormalize_globals(g_z, global_keys, manifest)
    dt_seq_seconds = _dt_norm_to_seconds(dt_seq_norm, manifest)

    y_pred_phys = _rollout(
        model,
        y0_phys=y0_phys,
        g_phys=g_phys,
        dt_seconds=dt_seq_seconds,
        infer_dtype=infer_dtype,
    )
    t_eval = np.cumsum(dt_seq_seconds, dtype=np.float64)

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_path = (
        output_dir
        / (
            f"autoregressive_rollout_{EVAL_SPLIT}_{shard_path.stem}"
            f"_sample-{sample_index}_start-{ROLLOUT_START_INDEX}_steps-{ROLLOUT_STEPS}"
            f"_run-{run_stamp}.png"
        )
    )
    source = f"{EVAL_SPLIT} {shard_path.stem}, sample {sample_index}, start {ROLLOUT_START_INDEX}"
    _plot(t_sec=t_eval, y_true=y_true_phys, y_pred=y_pred_phys, species_keys=species_keys, source=source,
          out_path=out_path)
    print(f"Saved: {out_path}")
    _print_errors(y_true_phys, y_pred_phys, eps=float(manifest["epsilon"]))


if __name__ == "__main__":
    main()
