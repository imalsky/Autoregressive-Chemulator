#!/usr/bin/env python3
"""
training_logs.py - Plot epoch-level training/validation losses from models/<run>/metrics.csv
and export a cleaned epoch-level metrics table.

A resumed run keeps each earlier segment as metrics.pre_restart[.N].csv (src/main.py); all
segments are merged in order (unsuffixed = 0, then N, then metrics.csv), so an epoch re-run
after a restart takes its latest values. LearningRateMonitor's epoch-less lr/* rows are
attached to the next epoch row of the same file (the LR at that epoch's start).

Run (the run directory is paths.work_dir of the config, testing/run_dir.py):
  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/training_logs.py

Saves:
  <run_dir>/metrics_cleaned.csv
  <run_dir>/plots/loss_curves.png

The legend gives the best val_loss and its epoch (the val_loss argmin over every segment, the epoch
best.ckpt holds, which export and the stage-2 handoff use) and the last value of each curve. When the
training rollout horizon changes (a stage-2 curriculum), it is drawn on the right axis: train_loss is
averaged over that horizon, val_loss always over training.rollout_steps from config.resolved.json.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import MaxNLocator, NullLocator

from run_dir import resolve_run_dir  # testing/ is this script's directory

# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

# Run/artifact selection: the run comes from AUTOCHEM_CONFIG_PATH (testing/run_dir.py).
METRICS_FILE: str = "metrics.csv"

# Output.
OUT_NAME: str = "loss_curves.png"
OUT_CLEAN_METRICS_NAME: str = "metrics_cleaned.csv"
PLOTS_SUBDIR: str = "plots"  # Relative to the run directory

# Plot behavior.
PLOT_COMPONENTS: bool = False  # Set True to also plot log10_mae and z_mse losses
SMOOTHING: int = 0             # 0 disables; otherwise moving-average window in epochs

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
_FIGSIZE = (6.5, 6.5)  # one square panel
_DPI = 300
# Colour per loss term (train solid, val dashed); the training rollout horizon K is dotted ink.
_LOSS_COLORS = {"total": _CYC[0], "log10_mae": _CYC[1], "z_mse": _CYC[2]}
_VAL_LS = "--"
_K_LS = ":"
_LEGEND_CORNERS = ("upper right", "upper left", "lower right", "lower left")
# Points sampled along each drawn segment when scoring a legend corner.
_SEGMENT_SAMPLES = 20

REQUIRED_PLOT_METRICS: tuple[str, ...] = (
    "train_loss",
    "val_loss",
    "train_loss_log10_mae",
    "val_loss_log10_mae",
    "train_loss_z_mse",
    "val_loss_z_mse",
)

# Earlier segments of a resumed run (src/main.py renames metrics.csv before each resume).
_SEGMENT_RE = re.compile(r"^metrics\.pre_restart(?:\.(\d+))?\.csv$")
# LearningRateMonitor columns (lr/pg1, lr/pg2, ...) are matched by this prefix.
_LR_PREFIX = "lr"

PREFERRED_METRIC_ORDER: tuple[str, ...] = (
    "step",
    "epoch_time_sec",
    "perf_train_step_ms",
    "perf_train_samples_per_s",
    "perf_train_tflops",
    "perf_val_time_s",
    "perf_peak_mem_gb",
    "lr",
    "grad_norm",
    "train_loss",
    "val_loss",
    "train_loss_log10_mae",
    "val_loss_log10_mae",
    "train_loss_z_mse",
    "val_loss_z_mse",
    "train_rollout_steps",
)


# =============================================================================
# IO
# =============================================================================

def _to_float(x: str) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _ordered_metric_names(metric_names: tuple[str, ...]) -> tuple[str, ...]:
    preferred: list[str] = []
    for want in PREFERRED_METRIC_ORDER:
        if want == _LR_PREFIX:
            preferred += sorted(n for n in metric_names if n.startswith(_LR_PREFIX))
        elif want in metric_names:
            preferred.append(want)
    remaining = sorted(name for name in metric_names if name not in preferred)
    return tuple(preferred + remaining)


def metrics_segments(run_dir: Path) -> list[Path]:
    """metrics.pre_restart[.N].csv by integer N (unsuffixed = 0), then metrics.csv; raise if none exist."""
    parts = []
    for p in run_dir.glob("metrics.pre_restart*.csv"):
        m = _SEGMENT_RE.match(p.name)
        if m:
            parts.append((int(m.group(1) or 0), p))
    files = [p for _, p in sorted(parts)]
    current = run_dir / METRICS_FILE
    if current.exists():
        files.append(current)
    if not files:
        raise FileNotFoundError(f"no {METRICS_FILE} or metrics.pre_restart*.csv in {run_dir}")
    return files


def load_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    """Epoch-level metrics merged over every segment of the run (see the module docstring)."""
    files = metrics_segments(run_dir)
    metric_names: list[str] = []
    by_epoch: dict[int, dict[str, float]] = {}
    for path in files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            metric_names += [n for n in (reader.fieldnames or []) if n and n != "epoch" and n not in metric_names]
            pending_lr: dict[str, float] = {}
            for r in reader:
                vals = {k: v for k, v in ((k, _to_float(x)) for k, x in r.items() if k and k != "epoch") if v is not None}
                e = _to_float(r.get("epoch", ""))
                if e is None:
                    pending_lr.update({k: v for k, v in vals.items() if k.startswith(_LR_PREFIX)})
                    continue
                row = by_epoch.setdefault(int(e), {})
                row.update(pending_lr)
                pending_lr = {}
                row.update(vals)

    if not by_epoch:
        raise ValueError(f"No epoch rows in: {[str(p) for p in files]}")

    epochs = np.array(sorted(by_epoch.keys()), dtype=np.int64)

    kept_metric_names = tuple(
        name
        for name in metric_names
        if name in REQUIRED_PLOT_METRICS
        or any(name in by_epoch[int(epoch)] for epoch in epochs)
    )
    ordered_metric_names = _ordered_metric_names(kept_metric_names)

    def col(name: str) -> np.ndarray:
        return np.array([by_epoch[int(epoch)].get(name, np.nan) for epoch in epochs], dtype=np.float64)

    data: dict[str, np.ndarray] = {"epoch": epochs}  # shape: (num_epochs,)
    for name in ordered_metric_names:
        data[name] = col(name)  # shape: (num_epochs,)
    return data


def _format_csv_value(value: float, *, integer: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    if integer:
        return str(int(round(float(value))))
    return f"{float(value):.16g}"


def save_clean_metrics_csv(metrics: dict[str, np.ndarray], out_path: Path) -> None:
    fieldnames = list(metrics.keys())
    integer_columns = {"epoch", "step"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        num_rows = int(metrics["epoch"].shape[0])
        for row_idx in range(num_rows):
            row = {
                name: _format_csv_value(
                    float(metrics[name][row_idx]),
                    integer=name in integer_columns,
                )
                for name in fieldnames
            }
            writer.writerow(row)

    print(f"Saved: {out_path}")


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    y2 = y.astype(np.float64, copy=True)
    y2[~np.isfinite(y2)] = np.nan
    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    ypad = np.pad(y2, (pad, w - 1 - pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def _sanitize_for_log(y: np.ndarray) -> np.ndarray:
    y2 = y.astype(np.float64, copy=True)
    y2[~np.isfinite(y2)] = np.nan
    y2[y2 <= 0.0] = np.nan
    return y2


def _format_last_metric(y: np.ndarray) -> str:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return "n/a"
    return f"{float(finite[-1]):.3e}"


def _best_epoch(epoch: np.ndarray, val: np.ndarray) -> tuple[int, float] | None:
    """(epoch, value) of the val_loss minimum, or None when no epoch has a finite positive value."""
    if not np.isfinite(val).any():
        return None
    i = int(np.nanargmin(val))
    return int(epoch[i]), float(val[i])


def _rollout_steps(run_dir: Path) -> int:
    """training.rollout_steps of the run (the validation horizon; metrics.csv has no column for it)."""
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    return int(cfg["training"]["rollout_steps"])


def _drawn_points(ax, line) -> np.ndarray:
    """Display coordinates of a line's points, plus _SEGMENT_SAMPLES points along each drawn segment
    (so a line crossing a legend counts, not only its vertices)."""
    xy = ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if line.get_linestyle() in ("None", "", " ") or len(xy) < 2:
        return xy
    t = np.linspace(0.0, 1.0, _SEGMENT_SAMPLES)[:, None, None]
    return (xy[:-1] * (1.0 - t) + xy[1:] * t).reshape(-1, 2)


def _legend(ax, handles: list, axes: list) -> None:
    """Legend in the corner covering the fewest plotted points of `axes`, one column then two;
    matplotlib's 'best' when every corner covers data. Never changes the limits."""
    fig = ax.figure
    for ncol in (1, 2):
        scores = {}
        for loc in _LEGEND_CORNERS:
            leg = ax.legend(handles=handles, loc=loc, ncol=ncol)
            fig.canvas.draw()
            x0, y0, x1, y1 = leg.get_window_extent().extents
            n = 0
            for a in axes:
                for line in a.lines:
                    xy = _drawn_points(a, line)
                    n += int(np.sum((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)))
            scores[loc] = n
        loc = min(scores, key=scores.get)
        if scores[loc] == 0:
            ax.legend(handles=handles, loc=loc, ncol=ncol)
            return
    ax.legend(handles=handles, loc="best")


# =============================================================================
# Plotting
# =============================================================================

def _resolve_clean_metrics_path(run_dir: Path) -> Path:
    return (run_dir / OUT_CLEAN_METRICS_NAME).resolve()


def _resolve_output_path(run_dir: Path) -> Path:
    return (run_dir / PLOTS_SUBDIR / OUT_NAME).resolve()


def plot_losses(m, out_path: Path, *, rollout_steps: int) -> None:
    """Loss curves on a log axis; best and last values in the legend; train K on a right axis when it varies."""
    epoch = m["epoch"]

    train = _sanitize_for_log(m["train_loss"])
    val = _sanitize_for_log(m["val_loss"])

    # Best and last from the raw values, before any smoothing: the best epoch is best.ckpt's.
    best = _best_epoch(epoch, val)
    best_txt = "n/a" if best is None else f"{best[1]:.3e} @ epoch {best[0]}"
    print(f"[best] val_loss {best_txt} (val_loss argmin over every metrics segment)")

    k = m.get("train_rollout_steps")
    k_fin = k[np.isfinite(k)] if k is not None else np.empty(0)
    k_varies = k_fin.size > 0 and float(k_fin.min()) != float(k_fin.max())
    # Two-line labels keep the legend narrow enough for a free corner.
    train_label = f"train_loss (last {_format_last_metric(train)})" + ("\nK: curriculum, right axis" if k_varies else "")
    val_k = f"K = {rollout_steps}; " if k_varies else ""
    val_label = f"val_loss (last {_format_last_metric(val)})\n{val_k}best {best_txt}"

    if SMOOTHING and SMOOTHING > 1:
        train = moving_average(train, SMOOTHING)
        val = moving_average(val, SMOOTHING)

    curves = [(train, _LOSS_COLORS["total"], "-", train_label), (val, _LOSS_COLORS["total"], _VAL_LS, val_label)]
    if PLOT_COMPONENTS:
        for term in ("log10_mae", "z_mse"):
            curves.append((_sanitize_for_log(m[f"train_loss_{term}"]), _LOSS_COLORS[term], "-", f"train_loss_{term}"))
            curves.append((_sanitize_for_log(m[f"val_loss_{term}"]), _LOSS_COLORS[term], _VAL_LS, f"val_loss_{term}"))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        handles = []
        for y, color, ls, label in curves:
            handles += ax.plot(epoch, y, color=color, ls=ls, label=label)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_box_aspect(1)
        axes = [ax]

        if k_varies:
            ax_k = ax.twinx()
            handles += ax_k.plot(epoch, k, color=_INK, ls=_K_LS, drawstyle="steps-post",
                                 label="train rollout K (right axis)")
            ax_k.set_ylabel("train rollout K")
            ax_k.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_k.yaxis.set_minor_locator(NullLocator())
            ax_k.set_box_aspect(1)
            axes.append(ax_k)

        # On the top-most axes, so no curve is drawn over it.
        _legend(axes[-1], handles, axes)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    run_dir = resolve_run_dir()
    segments = metrics_segments(run_dir)
    clean_metrics_path = _resolve_clean_metrics_path(run_dir)
    out_path = _resolve_output_path(run_dir)
    print(
        "[config] "
        f"run_dir={run_dir} metrics={[p.name for p in segments]} "
        f"clean_csv={clean_metrics_path} out={out_path}"
    )
    m = load_metrics(run_dir)
    save_clean_metrics_csv(m, clean_metrics_path)
    plot_losses(m, out_path, rollout_steps=_rollout_steps(run_dir))


if __name__ == "__main__":
    main()
