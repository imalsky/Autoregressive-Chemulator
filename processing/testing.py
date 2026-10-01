#!/usr/bin/env python3
"""
testing.py

2x2 panel figure of N_TRAJECTORIES = 2 randomly selected raw trajectories (groups), one per row;
row 0's window is anchored at the earliest feasible time, row 1's is not (preprocessing's mixture):
- left: log-x, log-y, every species of the raw trajectory with the window drawn from it
- right: linear-x, log-y, the same restricted to the window's time extent

Overlay style: raw outputs as RAW_STYLE (ink circles), window points as CHUNK_STYLE (red squares,
on top); every species in the same style; legend entries name the source and the window.

No argparse. The config file comes from AUTOCHEM_CONFIG_PATH (required);
plot settings are the globals below.
Outputs into: <repo_root>/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Shared preprocessing logic (processing/ is on sys.path when this script runs), so the
# chunks drawn here are built exactly as processing/preprocessing.py builds them.
from preprocessing import (
    PreCfg,
    _is_non_finite_raw_error,
    _is_non_positive_global_error,
    frozen_tail_cut,
    interp_loglog,
    leaf_dataset_index,
    list_raw_files,
    load_json,
    parse_precfg,
    pick_t_start,
    prepare_log_interp,
    read_globals_vector,
    read_species_matrix,
    read_time,
    sample_dt,
    validate_required_config_keys,
)


# =============================================================================
# USER SETTINGS (EDIT THESE)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The config file comes from AUTOCHEM_CONFIG_PATH (required; no default config).

# If None, rows cycle over the files matching config.preprocessing.raw_file_patterns under
# config.paths.raw_dir (row r uses file r mod n_files); otherwise every row uses this file.
RAW_FILE: Optional[Path] = None

# Number of trajectories, one row each of the 2x2 figure (log-x | window extent).
N_TRAJECTORIES = 2

# Random selection seed for trajectories
SEED = 1

# If set, overrides dt selection.
DT_OVERRIDE: Optional[float] = None

# Whether to apply preprocessing.drop_below rejection (strictly matches preprocessing).
# If True, any trajectory with any raw y < drop_below is skipped.
APPLY_DROP_BELOW = True

# Output folder/file
FIGURES_DIR = PROJECT_ROOT / "figures"
OUT_PNG = FIGURES_DIR / "chunk_vs_raw_panels.png"
DPI = 300

# Safety: max groups to attempt while searching for N_TRAJECTORIES valid ones
MAX_GROUP_TRIES = 5000
# Reject reasons listed in the failure message and the final summary.
TOP_REJECTS_SHOWN = 10

# Figure style: the user's scientific-plotting rules, inlined (repo scripts do not import them).
# Okabe-Ito cycle (yellow dropped), charcoal ink, serif text, ticks in on all sides, frameless
# 10 pt legend, zero x-margin and the style's 0.1 y-margin (limits from the data), 1.5 pt lines.
_CYC = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
_INK = "#2b2b2b"
_RED = "#CC3311"
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
_FIGSIZE = (10.7, 10.4)  # 2x2 square panels
# Raw (reference) vs window (derived): ink circles under red squares, data-point size 3.5 pt.
RAW_STYLE = {"ls": "", "marker": "o", "ms": 3.5, "color": _INK, "rasterized": True}
CHUNK_STYLE = {"ls": "", "marker": "s", "ms": 3.5, "color": _RED, "rasterized": True}
_LEGEND_CORNERS = ("upper right", "upper left", "lower right", "lower left")
# Points sampled along each drawn segment when scoring a legend corner.
_SEGMENT_SAMPLES = 20


# =============================================================================
# Chunk construction (preprocessing.sample_file logic for one trajectory)
# =============================================================================

def list_group_names(f_raw: h5py.File) -> List[str]:
    return [k for k in f_raw.keys() if isinstance(f_raw[k], h5py.Group)]


def build_raw_and_chunk_from_preprocessing_logic(
    raw: h5py.Group,
    cfg: PreCfg,
    rng: np.random.Generator,
    *,
    anchor_first: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      t_raw   : [Traw]
      y_raw   : [Traw, S]
      t_chunk : [n_steps] absolute times
      y_chunk : [n_steps, S] interpolated (floored at epsilon in log space)
      dt_s    : scalar dt used for the chunk

    anchor_first is preprocessing's per-sample choice: only the first of every
    samples_per_source_trajectory chunks is anchored at the earliest feasible time.

    Soft rejects raise RuntimeError("Rejected...: ..."); non-finite raw values and P/T that are
    not > 0 raise preprocessing's ValueError; raw-schema errors raise KeyError/TypeError/ValueError.
    """
    leaf_index = leaf_dataset_index(raw)

    t_raw = read_time(raw, time_key=cfg.time_key, leaf_index=leaf_index)
    T = int(t_raw.shape[0])

    y_raw = read_species_matrix(raw, t_len=T, species_vars=cfg.species_variables, leaf_index=leaf_index)
    # Globals are not plotted; reading them rejects bad P/T exactly as preprocessing does.
    read_globals_vector(raw, global_vars=cfg.global_variables, leaf_index=leaf_index)

    # Apply preprocessing.drop_below rejection (optional; default True to match preprocessing)
    if APPLY_DROP_BELOW:
        if np.any(y_raw < float(cfg.drop_below)):
            raise RuntimeError(f"Rejected by drop_below: min(y_raw)={float(np.min(y_raw)):.3e} < {cfg.drop_below:.3e}")

    # Valid mask (mirrors preprocessing.sample_file): positive times, at least 2 of them, cut at a
    # frozen raw tail (frozen_tail_cut).
    valid = t_raw > 0
    if int(np.count_nonzero(valid)) < 2:
        raise RuntimeError("Rejected: fewer than 2 positive times (too_few_valid)")
    cut = frozen_tail_cut(y_raw, epsilon=cfg.epsilon)
    if cut is not None:
        valid &= np.arange(T) <= cut

    t_valid = t_raw[valid]
    y_valid = y_raw[valid, :]

    # Sample dt (preprocessing.sample_dt unless DT_OVERRIDE is set)
    dt_s = float(DT_OVERRIDE) if DT_OVERRIDE is not None else sample_dt(cfg, rng)

    # Choose chunk start
    t_start = pick_t_start(
        t_raw=t_raw,
        t_valid=t_valid,
        dt_s=dt_s,
        n_steps=cfg.n_steps,
        t_min=cfg.t_min,
        rng=rng,
        anchor_first=anchor_first,
        max_attempts=cfg.max_chunk_attempts_per_source,
    )
    if t_start is None:
        raise RuntimeError("Rejected: could not fit a chunk (no_fit_chunk)")

    t_chunk = float(t_start) + np.arange(cfg.n_steps, dtype=np.float64) * float(dt_s)

    # Interpolate in log-time/log-y (preprocessing.interp_loglog)
    log_t_valid = np.log10(t_valid)
    log_t_chunk = np.log10(t_chunk)
    i0, i1, w = prepare_log_interp(log_t_valid, log_t_chunk)
    y_chunk = interp_loglog(y_valid, i0=i0, i1=i1, w=w, epsilon=cfg.epsilon)

    if not np.all(np.isfinite(y_chunk)):
        raise RuntimeError("Rejected: interp produced non-finite values")

    return t_raw, y_raw, t_chunk, y_chunk, float(dt_s)


# =============================================================================
# Plotting
# =============================================================================

def _drawn_points(ax, line) -> np.ndarray:
    """Display coordinates of a line's points, plus _SEGMENT_SAMPLES points along each drawn segment
    (so a line crossing a legend counts, not only its vertices)."""
    xy = ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if line.get_linestyle() in ("None", "", " ") or len(xy) < 2:
        return xy
    t = np.linspace(0.0, 1.0, _SEGMENT_SAMPLES)[:, None, None]
    return (xy[:-1] * (1.0 - t) + xy[1:] * t).reshape(-1, 2)


def _legend(ax) -> None:
    """Legend in the corner covering the fewest plotted points, one column then two; matplotlib's
    'best' when every corner covers data. Never changes the limits."""
    fig = ax.figure
    for ncol in (1, 2):
        scores = {}
        for loc in _LEGEND_CORNERS:
            leg = ax.legend(loc=loc, ncol=ncol)
            fig.canvas.draw()
            x0, y0, x1, y1 = leg.get_window_extent().extents
            n = 0
            for line in ax.lines:
                xy = _drawn_points(ax, line)
                n += int(np.sum((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)))
            scores[loc] = n
        loc = min(scores, key=scores.get)
        if scores[loc] == 0:
            ax.legend(loc=loc, ncol=ncol)
            return
    ax.legend(loc="best")


def style_axes(ax, *, xlog: bool) -> None:
    ax.set_yscale("log")
    ax.set_xscale("log" if xlog else "linear")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mixing ratio")
    ax.set_box_aspect(1)


def plot_overlay_all_species(
    ax,
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    t_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    legend: bool,
    dt_s: float,
    source: str,
    t_window: Optional[Tuple[float, float]],
) -> None:
    """Raw and window points of every species, limits from the plotted data (t_window: only raw outputs
    inside it are drawn, so the window sets the x extent).

    With `legend`, the panel carries the legend (source, dt, start, raw outputs inside the window)
    and the full counts are printed.
    """
    # Raw outputs inside the window: few means the window is mostly interpolation.
    n_inside = int(np.count_nonzero((t_raw >= t_chunk[0]) & (t_raw <= t_chunk[-1])))
    raw_label = f"raw {source}"
    chunk_label = f"window: dt = {dt_s:.3g} s, start {float(t_chunk[0]):.3g} s\n{n_inside} raw outputs inside"
    if legend:
        print(f"{source}: {int(np.sum(t_raw > 0))} raw points per species x {y_raw.shape[1]} species; window of "
              f"{len(t_chunk)} points, dt={dt_s:.6g} s, start={float(t_chunk[0]):.6g} s, {n_inside} raw outputs inside")

    did_raw_label = False
    did_chunk_label = False

    S = int(y_raw.shape[1])
    for j in range(S):
        yr = y_raw[:, j].astype(float, copy=False)
        yc = y_chunk[:, j].astype(float, copy=False)

        m_raw = (t_raw > 0) & (yr > 0)
        m_chunk = (t_chunk > 0) & (yc > 0)
        if t_window is not None:
            m_raw &= (t_raw >= t_window[0]) & (t_raw <= t_window[1])

        ax.plot(t_raw[m_raw], yr[m_raw], label=(None if did_raw_label else raw_label), **RAW_STYLE)
        did_raw_label = True
        ax.plot(t_chunk[m_chunk], yc[m_chunk], label=(None if did_chunk_label else chunk_label), **CHUNK_STYLE)
        did_chunk_label = True

    if legend:
        _legend(ax)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg_override = os.environ.get("AUTOCHEM_CONFIG_PATH", "").strip()
    if not cfg_override:
        raise RuntimeError(
            "AUTOCHEM_CONFIG_PATH is not set. Point it at an explicit config file, e.g.:\n"
            "  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u processing/testing.py"
        )
    config_path = Path(cfg_override).expanduser().resolve()
    # Same strict parsing as preprocessing (unknown or missing keys hard-fail; no defaults).
    cfg_dict = load_json(config_path)
    validate_required_config_keys(cfg_dict)
    cfg = parse_precfg(cfg_dict, cfg_path=config_path)
    if DT_OVERRIDE is not None and DT_OVERRIDE <= 0:
        raise ValueError(f"DT_OVERRIDE must be > 0, got {DT_OVERRIDE}")

    if RAW_FILE is not None:
        raw_files = [RAW_FILE if RAW_FILE.is_absolute() else (PROJECT_ROOT / RAW_FILE).resolve()]
    else:
        raw_files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
        if not raw_files:
            raise FileNotFoundError(f"No HDF5 files found under raw_dir={cfg.raw_dir} for patterns={cfg.raw_file_patterns}")
    for raw_file in raw_files:
        if not raw_file.exists():
            raise FileNotFoundError(f"RAW_FILE not found: {raw_file}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    samples: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    used: set[Tuple[str, str]] = set()
    reject_reasons: Dict[str, int] = {}

    tries = 0
    while len(samples) < N_TRAJECTORIES and tries < MAX_GROUP_TRIES:
        r = len(samples)
        raw_file = raw_files[r % len(raw_files)]
        # Row r mirrors preprocessing's mixture: one anchored chunk per samples_per_source_trajectory.
        anchor_first = r % int(cfg.samples_per_source_trajectory) == 0
        with h5py.File(raw_file, "r") as f_raw:
            group_names = list_group_names(f_raw)
            if not group_names:
                raise RuntimeError(f"No groups found at root of raw file: {raw_file}")
            while tries < MAX_GROUP_TRIES:
                tries += 1
                gname = group_names[int(rng.integers(0, len(group_names)))]
                if (raw_file.name, gname) in used:
                    continue
                used.add((raw_file.name, gname))

                raw_grp = f_raw[gname]
                try:
                    t_raw, y_raw, t_chunk, y_chunk, dt_s = build_raw_and_chunk_from_preprocessing_logic(
                        raw_grp, cfg, rng, anchor_first=anchor_first
                    )
                except ValueError as e:
                    # preprocessing's split: non-finite raw values and non-positive P/T are soft
                    # rejects; any other ValueError (like KeyError/TypeError) is a raw-schema error.
                    if _is_non_finite_raw_error(e):
                        key = "non_finite"
                    elif _is_non_positive_global_error(e):
                        key = "non_positive_global"
                    else:
                        raise
                    reject_reasons[key] = reject_reasons.get(key, 0) + 1
                    continue
                except RuntimeError as e:  # this script's own soft rejects
                    key = str(e).split(":")[0].strip()
                    reject_reasons[key] = reject_reasons.get(key, 0) + 1
                    continue

                samples.append((raw_file.name, gname, t_raw, y_raw, t_chunk, y_chunk, dt_s))
                break

    def top_rejects() -> str:
        ranked = sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:TOP_REJECTS_SHOWN]
        return ", ".join(f"{k}={v}" for k, v in ranked)

    if len(samples) < N_TRAJECTORIES:
        raise RuntimeError(
            f"Could only find {len(samples)} valid trajectories (wanted {N_TRAJECTORIES}) after {tries} tries. "
            f"Top {TOP_REJECTS_SHOWN} reject reasons: {top_rejects()}"
        )

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(N_TRAJECTORIES, 2, figsize=_FIGSIZE, squeeze=False, constrained_layout=True)
        for r, (_fname, gname, t_raw, y_raw, t_chunk, y_chunk, dt_s) in enumerate(samples):
            for col, xlog in enumerate((True, False)):
                ax = axes[r][col]
                style_axes(ax, xlog=xlog)
                plot_overlay_all_species(
                    ax,
                    t_raw,
                    y_raw,
                    t_chunk,
                    y_chunk,
                    # One legend per row, on the left panel (the right one draws the same styles).
                    legend=xlog,
                    dt_s=dt_s,
                    source=gname,
                    # Right column: x restricted to the window's extent.
                    t_window=None if xlog else (float(t_chunk[0]), float(t_chunk[-1])),
                )
        fig.savefig(OUT_PNG, dpi=DPI)
        plt.close(fig)

    print(f"Wrote: {OUT_PNG.resolve()}")
    print("Rows (file:trajectory):", [f"{s[0]}:{s[1]}" for s in samples])
    if reject_reasons:
        print(f"Top {TOP_REJECTS_SHOWN} reject reasons:", top_rejects())

if __name__ == "__main__":
    main()
