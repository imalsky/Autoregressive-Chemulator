#!/usr/bin/env python3
"""
testing.py

3x2 panel figure:
- 3 rows: 3 different randomly selected raw trajectories (groups)
- Left column: log-x + log-y (loglog), overlay all species (raw vs chunk)
- Right column: linear-x + log-y (semilog-y), overlay all species (raw vs chunk),
  with x-axis restricted to the extent of the blue (regridded) points.

Overlay style:
- raw: red circles ('ro', ms=8)
- chunk: blue squares ('bs', ms=5)
- alpha=0.35
- legend entries for Raw / Chunk

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

# Shared preprocessing logic (processing/ is on sys.path when this script runs), so the
# chunks drawn here are built exactly as processing/preprocessing.py builds them.
from preprocessing import (
    PreCfg,
    _is_non_finite_raw_error,
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

# If None, selects first matching file under config.paths.raw_dir using
# config.preprocessing.raw_file_patterns.
RAW_FILE: Optional[Path] = None

# Number of rows (trajectories) and layout
N_ROWS = 3
N_COLS = 2

# Random selection seed for trajectories
SEED = 1

# Chunk start semantics (preprocessing anchors only the first chunk per trajectory)
USE_ANCHOR_FIRST_CHUNK = True

# If set, overrides dt selection.
DT_OVERRIDE: Optional[float] = None

# Whether to apply preprocessing.drop_below rejection (strictly matches preprocessing).
# If True, any trajectory with any raw y < drop_below is skipped.
APPLY_DROP_BELOW = True

# Output folder/file
FIGURES_DIR = PROJECT_ROOT / "figures"
OUT_PNG = FIGURES_DIR / "chunk_vs_raw_panels.png"
DPI = 250

# Safety: max groups to attempt while searching for N_ROWS valid ones
MAX_GROUP_TRIES = 5000


# =============================================================================
# Chunk construction (preprocessing.sample_file logic for one trajectory)
# =============================================================================

def list_group_names(f_raw: h5py.File) -> List[str]:
    return [k for k in f_raw.keys() if isinstance(f_raw[k], h5py.Group)]


def build_raw_and_chunk_from_preprocessing_logic(
    raw: h5py.Group,
    cfg: PreCfg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      t_raw   : [Traw]
      y_raw   : [Traw, S]
      t_chunk : [n_steps] absolute times
      y_chunk : [n_steps, S] interpolated (floored at epsilon in log space)
      dt_s    : scalar dt used for the chunk

    Soft rejects raise RuntimeError("Rejected...: ..."); non-finite raw values raise
    preprocessing's ValueError; raw-schema errors raise KeyError/TypeError/ValueError.
    """
    leaf_index = leaf_dataset_index(raw)

    t_raw = read_time(raw, time_key=cfg.time_key, leaf_index=leaf_index)
    T = int(t_raw.shape[0])

    y_raw = read_species_matrix(raw, t_len=T, species_vars=cfg.species_variables, leaf_index=leaf_index)
    # Globals are not plotted; reading them rejects non-finite P/T exactly as preprocessing does.
    read_globals_vector(raw, global_vars=cfg.global_variables, leaf_index=leaf_index)

    # Apply preprocessing.drop_below rejection (optional; default True to match preprocessing)
    if APPLY_DROP_BELOW:
        if np.any(y_raw < float(cfg.drop_below)):
            raise RuntimeError(f"Rejected by drop_below: min(y_raw)={float(np.min(y_raw)):.3e} < {cfg.drop_below:.3e}")

    # Valid mask (mirrors preprocessing.sample_file): positive times, at least 2 of them.
    valid = t_raw > 0
    if int(np.count_nonzero(valid)) < 2:
        raise RuntimeError("Rejected: fewer than 2 positive times (too_few_valid)")

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
        anchor_first=USE_ANCHOR_FIRST_CHUNK,
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

def style_axes(ax, *, xlog: bool) -> None:
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_yscale("log")
    ax.set_xscale("log" if xlog else "linear")
    ax.set_xlabel("Time", labelpad=6)
    ax.set_ylabel("Mixing ratio", labelpad=6)


def plot_overlay_all_species(
    ax,
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    t_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    legend: bool,
    species_count: int,
    dt_s: float,
) -> None:
    raw_pts_per_species = int(np.sum(t_raw > 0))
    chunk_pts = int(len(t_chunk))

    raw_label = f"Raw ({raw_pts_per_species} pts/species, N={species_count})"
    chunk_label = f"Chunk ({chunk_pts} pts, dt={dt_s:.3g} s)"

    did_raw_label = False
    did_chunk_label = False

    S = int(y_raw.shape[1])
    for j in range(S):
        yr = y_raw[:, j].astype(float, copy=False)
        yc = y_chunk[:, j].astype(float, copy=False)

        m_raw = (t_raw > 0) & (yr > 0)
        m_chunk = (t_chunk > 0) & (yc > 0)

        ax.plot(
            t_raw[m_raw],
            yr[m_raw],
            "ro",
            ms=8,
            alpha=0.35,
            label=(raw_label if (legend and not did_raw_label) else None),
            rasterized=True,
        )
        did_raw_label = True

        ax.plot(
            t_chunk[m_chunk],
            yc[m_chunk],
            "bs",
            ms=5,
            alpha=0.35,
            label=(chunk_label if (legend and not did_chunk_label) else None),
            rasterized=True,
        )
        did_chunk_label = True

    if legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)


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
        raw_file = RAW_FILE if RAW_FILE.is_absolute() else (PROJECT_ROOT / RAW_FILE).resolve()
    else:
        raw_files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
        if not raw_files:
            raise FileNotFoundError(f"No HDF5 files found under raw_dir={cfg.raw_dir} for patterns={cfg.raw_file_patterns}")
        raw_file = raw_files[0]
    if not raw_file.exists():
        raise FileNotFoundError(f"RAW_FILE not found: {raw_file}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    with h5py.File(raw_file, "r") as f_raw:
        group_names = list_group_names(f_raw)
        if not group_names:
            raise RuntimeError(f"No groups found at root of raw file: {raw_file}")

        samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        used: set[str] = set()
        reject_reasons: Dict[str, int] = {}

        tries = 0
        while len(samples) < N_ROWS and tries < MAX_GROUP_TRIES:
            tries += 1
            gname = group_names[int(rng.integers(0, len(group_names)))]
            if gname in used:
                continue
            used.add(gname)

            raw_grp = f_raw[gname]
            try:
                t_raw, y_raw, t_chunk, y_chunk, dt_s = build_raw_and_chunk_from_preprocessing_logic(raw_grp, cfg, rng)
            except ValueError as e:
                # preprocessing's split: non-finite raw values are a soft reject; any other
                # ValueError (like KeyError/TypeError) is a raw-schema error and propagates.
                if not _is_non_finite_raw_error(e):
                    raise
                reject_reasons["non_finite"] = reject_reasons.get("non_finite", 0) + 1
                continue
            except RuntimeError as e:  # this script's own soft rejects
                key = str(e).split(":")[0].strip()
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                continue

            samples.append((gname, t_raw, y_raw, t_chunk, y_chunk, dt_s))

        if len(samples) < N_ROWS:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:12])
            raise RuntimeError(
                f"Could only find {len(samples)} valid trajectories (wanted {N_ROWS}) after {tries} tries. "
                f"Top reject reasons: {summary}"
            )

        fig, axes = plt.subplots(
            N_ROWS,
            N_COLS,
            figsize=(14.0, 4.2 * N_ROWS),
            squeeze=False,
        )

        for r, (gname, t_raw, y_raw, t_chunk, y_chunk, dt_s) in enumerate(samples):
            # Left: log-x (and log-y)
            axL = axes[r][0]
            style_axes(axL, xlog=True)
            axL.set_title(f"{gname} (log-x)", pad=10)
            plot_overlay_all_species(
                axL,
                t_raw,
                y_raw,
                t_chunk,
                y_chunk,
                legend=True,
                species_count=len(cfg.species_variables),
                dt_s=dt_s,
            )

            # Right: linear-x (log-y) BUT x-range exactly equals blue-dot extent
            axR = axes[r][1]
            style_axes(axR, xlog=False)
            axR.set_title(f"{gname} (linear-x; xlim=chunk extent)", pad=10)
            plot_overlay_all_species(
                axR,
                t_raw,
                y_raw,
                t_chunk,
                y_chunk,
                legend=True,
                species_count=len(cfg.species_variables),
                dt_s=dt_s,
            )
            axR.set_xlim(float(t_chunk[0]), float(t_chunk[-1]))

        fig.suptitle(
            f"Raw vs Generated Chunk Overlay (dt_sampling={cfg.dt_sampling}, n_steps={cfg.n_steps}, eps={cfg.epsilon:.1e})\n"
            f"Raw file: {raw_file.name}",
            fontsize=11,
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.965])

        fig.savefig(OUT_PNG, dpi=DPI)
        plt.close(fig)

        print(f"Wrote: {OUT_PNG.resolve()}")
        print("Rows (trajectories):", [s[0] for s in samples])
        if reject_reasons:
            top = ", ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:10])
            print("Top reject reasons:", top)


if __name__ == "__main__":
    main()
