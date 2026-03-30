#!/usr/bin/env python3
"""
training_logs.py - Plot epoch-level training/validation losses from models/<run>/metrics.csv
and export a cleaned epoch-level metrics table.

Assumes repo layout:
  <repo>/
    src/
    testing/  (this file)
    testing/science.mplstyle
    models/<run>/metrics.csv

Saves:
  <run_dir>/metrics_cleaned.csv
  <run_dir>/plots/loss_curves.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

# Run/artifact selection (main knobs).
RUN_DIR: str = "models_old/v3"  # Absolute path or repo-relative path
METRICS_FILE: str = "metrics.csv"

# Output.
OUT_NAME: str = "loss_curves.png"
OUT_CLEAN_METRICS_NAME: str = "metrics_cleaned.csv"
PLOTS_SUBDIR: str = "plots"  # Relative to RUN_DIR

# Plot behavior.
PLOT_COMPONENTS: bool = False  # Set True to also plot log10_mae and z_mse losses
SMOOTHING: int = 0             # 0 disables; otherwise moving-average window in epochs

# Style candidates (first match wins).
STYLE_CANDIDATES: tuple[Path | str, ...] = (
    ROOT / "science.mplstyle",
    ROOT / "testing" / "science.mplstyle",
    "science.mplstyle",
)

REQUIRED_PLOT_METRICS: tuple[str, ...] = (
    "train_loss",
    "val_loss",
    "train_loss_log10_mae",
    "val_loss_log10_mae",
    "train_loss_z_mse",
    "val_loss_z_mse",
)

PREFERRED_METRIC_ORDER: tuple[str, ...] = (
    "step",
    "epoch_time_sec",
    "lr",
    "grad_norm",
    "train_loss",
    "val_loss",
    "train_loss_log10_mae",
    "val_loss_log10_mae",
    "train_loss_z_mse",
    "val_loss_z_mse",
    "train_rollout_steps",
    "train_skip_steps",
    "train_detach_between_steps",
)


def _resolve_repo_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p.resolve()


# =============================================================================
# Style
# =============================================================================

def apply_style() -> None:
    for s in STYLE_CANDIDATES:
        try:
            if isinstance(s, Path):
                if s.exists():
                    plt.style.use(str(s))
                    return
            else:
                plt.style.use(s)
                return
        except Exception:
            pass


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
    preferred = [name for name in PREFERRED_METRIC_ORDER if name in metric_names]
    remaining = sorted(name for name in metric_names if name not in PREFERRED_METRIC_ORDER)
    return tuple(preferred + remaining)


def load_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        metric_names = tuple(name for name in (reader.fieldnames or []) if name and name != "epoch")

    if not rows:
        raise ValueError(f"No rows in: {path}")

    # Merge rows per epoch because train/val metrics may be logged on separate rows.
    by_epoch: dict[int, dict[str, float | None]] = {}
    for r in rows:
        e = _to_float(r.get("epoch", ""))
        if e is None:
            continue
        epoch = int(e)
        metrics = by_epoch.setdefault(epoch, {k: None for k in metric_names})
        for k in metric_names:
            v = _to_float(r.get(k, ""))
            if v is not None:
                metrics[k] = v

    if not by_epoch:
        raise ValueError(f"No epoch rows in: {path}")

    epochs = np.array(sorted(by_epoch.keys()), dtype=np.int64)

    kept_metric_names = tuple(
        name
        for name in metric_names
        if name in REQUIRED_PLOT_METRICS
        or any(by_epoch[int(epoch)][name] is not None for epoch in epochs)
    )
    ordered_metric_names = _ordered_metric_names(kept_metric_names)

    def col(name: str) -> np.ndarray:
        return np.array(
            [
                by_epoch[int(epoch)][name] if by_epoch[int(epoch)][name] is not None else np.nan
                for epoch in epochs
            ],
            dtype=np.float64,
        )

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


def _format_final_metric(y: np.ndarray) -> str:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return "n/a"
    return f"{float(finite[-1]):.3e}"


# =============================================================================
# Plotting
# =============================================================================

def _resolve_metrics_path(run_dir: Path) -> Path:
    return (run_dir / METRICS_FILE).resolve()


def _resolve_clean_metrics_path(run_dir: Path) -> Path:
    return (run_dir / OUT_CLEAN_METRICS_NAME).resolve()


def _resolve_output_path(run_dir: Path) -> Path:
    return (run_dir / PLOTS_SUBDIR / OUT_NAME).resolve()


def plot_losses(m, out_path: Path, *, run_name: str) -> None:
    epoch = m["epoch"]

    train = _sanitize_for_log(m["train_loss"])
    val = _sanitize_for_log(m["val_loss"])

    if SMOOTHING and SMOOTHING > 1:
        train = moving_average(train, SMOOTHING)
        val = moving_average(val, SMOOTHING)

    train_label = f"train_loss (final={_format_final_metric(train)})"
    val_label = f"val_loss (final={_format_final_metric(val)})"

    fig, ax = plt.subplots(figsize=(6, 6))


    ax.plot(
        epoch,
        val,
        label=val_label,
        color="black",
        linestyle=(0, (7, 8)),
        linewidth=2,
        alpha=0.95,
        dash_capstyle="round",
    )
    ax.plot(
        epoch,
        train,
        label=train_label,
        color="black",
        linestyle="solid",
        linewidth=2,
        alpha=1,
        solid_capstyle="round",
    )


    if PLOT_COMPONENTS:
        tlm = _sanitize_for_log(m["train_loss_log10_mae"])
        vlm = _sanitize_for_log(m["val_loss_log10_mae"])
        tzm = _sanitize_for_log(m["train_loss_z_mse"])
        vzm = _sanitize_for_log(m["val_loss_z_mse"])

        ax.plot(epoch, tlm, linestyle="solid", label="train_log10_mae", color="tab:red", linewidth=2.0, alpha=0.8)
        ax.plot(epoch, vlm, linestyle="dashed", label="val_log10_mae", color="tab:red", linewidth=2.0, alpha=0.9)
        ax.plot(epoch, vzm, linestyle="dashed", label="val_z_mse", color="tab:blue", linewidth=2.0, alpha=0.9)
        ax.plot(epoch, tzm, linestyle="solid", label="train_z_mse", color="tab:blue", linewidth=2.0, alpha=0.8)


    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_box_aspect(1)  # square plot area

    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    apply_style()
    run_dir = _resolve_repo_path(RUN_DIR)
    metrics_path = _resolve_metrics_path(run_dir)
    clean_metrics_path = _resolve_clean_metrics_path(run_dir)
    out_path = _resolve_output_path(run_dir)
    print(
        "[config] "
        f"run_dir={run_dir} metrics={metrics_path} "
        f"clean_csv={clean_metrics_path} out={out_path}"
    )
    m = load_metrics_csv(metrics_path)
    save_clean_metrics_csv(m, clean_metrics_path)
    plot_losses(m, out_path, run_name=run_dir.name)


if __name__ == "__main__":
    main()
