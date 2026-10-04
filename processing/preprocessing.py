#!/usr/bin/env python3
"""preprocessing.py

Preprocess raw HDF5 trajectories, saved on a fixed log-spaced output grid (current files: t=0,
then 1e-3..1e8 s, 100 points), into fixed-length uniform-dt windows, shard to NPZ, compute
normalization, and write final normalized shards. The raw dt is non-uniform, so every window is
resampled (log-log interpolation); each group's own time vector is used, so a file on another grid
works too (its grid is recorded per file in preprocessing_summary.json).

Outputs (under paths.processed_dir):
  normalization.json          normalization manifest read by training and export: methods and
                              train-split statistics, plus units, trained-domain records
                              (window start times, (T, P) coverage, tracked-species sum) and the
                              std-floor record
  preprocessing_summary.json  build record: per-file rows (size, groups, rejects, windows, shards,
                              raw time grid), split definition, (T, P) coverage table, window-start
                              and dt histograms, the config blocks used, code/library versions
  train/shard_*.npz
  validation/shard_*.npz
  test/shard_*.npz

Temporary stage (under paths.processed_dir):
  _tmp_physical/<split>/shard_*_physical.npz   (each deleted once its final shard is written)
  _tmp_final/                                  (the new dataset, built before anything is replaced)
  _tmp_old/                                    (the previous dataset while it is being swapped out)
  All three are removed on success. A failed run leaves them in place; the next run wipes
  them (preprocessing.overwrite=true) or refuses (false). The previous dataset stays intact
  until the new one is complete: only preprocessing-owned entries are swapped, with
  normalization.json (the launchers' "dataset complete" marker) moved out first and in last.

Lock: <processed_dir>.lock, a persistent 0-byte sibling file, is flocked exclusively for the whole
run so two preprocessing runs (or leftover workers of a killed run) never share a processed_dir,
and training (src/main.py, which holds it shared from reading normalization.json until both splits
are preloaded) or the rollout-error evaluation never reads a half-replaced dataset. Either side
fails at once while the other holds it.

Shard schema:
  Physical shards (temporary):
    y_mat   : [N, n_steps, S] float32  (physical units)
    globals : [N, G]          float32  (physical units)
    dt_mat  : [N, n_steps-1]  float32  (physical dt, constant within each sample)

  Final shards:
    y_mat       : [N, n_steps, S] float32  (z-space, log-standard)
    globals     : [N, G]          float32  (normalized globals)
    dt_norm_mat : [N, n_steps-1]  float32  (normalized dt in [0, 1])

Frozen raw tails: a trajectory whose raw state stopped changing in mid-evolution and was then
repeated row for row (frozen_tail_cut) is cut after its last changing row, so its windows come
only from the evolving part; the group is kept, or rejected as frozen_tail_no_window if no window
fits before the cut. Converged repeated tails are kept. preprocessing_summary.json frozen_tail
records the threshold and the per-split counts.

Randomness: split membership is a hash of (SPLIT_SEED, file basename, group name), independent of
preprocessing.seed. Window draws (dt, start time) use one generator per trajectory, seeded with
[preprocessing.seed, a hash of the group name], so they do not depend on the file's position in
the file list or on which other groups were kept. The per-file generator (seeded with the file
name) only shuffles the group pool.

This script is intentionally strict:
- Configuration must use the canonical schema (the file named by AUTOCHEM_CONFIG_PATH).
- Dataset names must match exactly (no implicit renaming).
- Globals must be scalar per trajectory: a non-scalar P/T fails the run as a schema
  mismatch; a non-finite one (non_finite) or one that is not > 0 as float32
  (non_positive_global) rejects only that trajectory group, counted in the summary.

Run:
  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u processing/preprocessing.py
"""

from __future__ import annotations

import ctypes
import datetime
import fcntl
import hashlib
import json
import logging
import math
import multiprocessing
import os
import platform
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# When run as a script, cap native BLAS/OpenMP threads to 1 BEFORE numpy/h5py load their
# backends: the fan-out runs one worker process per raw file (up to preprocessing.num_workers),
# and each worker would otherwise start its own BLAS thread pool and oversubscribe the cores.
# Hard-assigned, so thread counts exported for training do not leak into the fan-out. Modules
# that import this file (processing/testing.py) keep their own thread settings.
if __name__ == "__main__":
    for _thread_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[_thread_var] = "1"

# These imports must stay below the thread-cap loop: the BLAS/OpenMP backends
# read the env vars above at import time.
import h5py
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    """Initialize root logging at the named level (e.g. "INFO"); raise on a bad level."""
    lvl = getattr(logging, str(level).upper().strip(), None)
    if lvl is None:
        raise ValueError(f"Unsupported log level: {level}")
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


log = logging.getLogger(__name__)


# Fixed global set, in this order (spec.md section 5, items 4-6). This standalone script
# imports nothing from src/, so it keeps its own copy of src/model.py:_REQUIRED_GLOBALS.
_REQUIRED_GLOBALS: Tuple[str, str] = ("P", "T")
# Physical units of the raw inputs, written to normalization.json. The raw files carry no unit
# attributes and preprocessing applies no conversion, so these declare the VULCAN source's units
# (re-checked 2026-10-01; species convention and helium: docs/DECISIONS.md open item 25).
UNITS: Dict[str, str] = {
    "P": "dyn/cm^2 (barye)",
    "T": "K",
    "dt": "s",
    "time": "s (since the raw t=0)",
    "species": (
        "mixing ratio relative to the whole gas; training compositions contain no He: y_i = n_i / "
        "n_total over all gas species of the generator run, untracked species included, not "
        "renormalized over the tracked species; the training initial gas held only the tracked "
        "species (their sum is 1 at t=0 and at most 1 afterwards; train-split distribution of the "
        "sum: normalization.json species_sum); He-bearing compositions are outside the validated "
        "domain; model outputs use the same denominator"
    ),
}
# Split membership: a fixed constant, not preprocessing.seed, so a sampling replicate (another
# preprocessing.seed) keeps the same train/validation/test trajectories. 1234 is the historical
# preprocessing.seed, so today's splits are unchanged.
SPLIT_SEED = 1234
# split_for_trajectory keys on 64 bits of SHA-1, mapped to a uniform in (0, 1) as (u + 0.5) / 2**64.
_SPLIT_HASH_BYTES = 8
SPLIT_KEY_DEFINITION = (
    "u = int.from_bytes(sha1(f'{SPLIT_SEED}:{file basename}:{group name}').digest()[:8], 'big'); "
    "x = (u + 0.5) / 2**64, uniform in (0, 1); test if x < test_fraction, validation if "
    "x < test_fraction + val_fraction, else train"
)
# Generators are seeded with [preprocessing.seed, one 64-bit word of SHA-1 of an identity]: the file
# name (pool shuffle) or the group name (window draws). The same word is the group-name digest of
# the cross-file duplicate check.
_SEED_WORD_BYTES = 8
# (T, log10 P) coverage cells, floor-based and open-ended (no fixed box: the synthetic smoke data
# reaches log10 P = -2). 100 K x 1 dex resolves the drop_below skew (>= 144 kept groups per cell on
# the 4-file set, audit R1-018).
_COVERAGE_T_BIN_K = 100.0
_COVERAGE_LOG10P_BIN_DEX = 1.0
# Histograms of the written windows' log10 start time and log10 dt (floor-based, open-ended bins):
# 1 dex spans the 11-decade raw grid in 11 bins, 0.5 dex the 4-decade dt range in 8.
_TSTART_HIST_BIN_DEX = 1.0
_DT_HIST_BIN_DEX = 0.5
# Train-split distribution of the tracked-species sum per state: a fixed histogram on [0, 1]
# (sums above 1 from float32 rounding land in the last bin) with this resolution, read at these
# cumulative levels (the low tail, plus the median).
_SPECIES_SUM_HIST_BINS = 10_000
_SPECIES_SUM_QUANTILE_LEVELS: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.5)
# Best-effort git provenance in the summary: a git call slower than this is recorded as null.
_GIT_TIMEOUT_S = 10.0
# Frozen raw tails (frozen_tail_cut; docs/DECISIONS.md open item 17): a repeated-state tail is cut when
# the step into it still moves a species by more than this (dex). Species >= 1e-12 never exceed
# 7.2e-4 dex there on the 4 raw files (audit R2-006); recheck on new data.
_FROZEN_TAIL_MAX_PRECHANGE_DEX = 1e-3
_SPLITS: Tuple[str, str, str] = ("train", "validation", "test")
# The manifest keys that define the z-space; their canonical-JSON sha256 is the normalization
# fingerprint (norm_sha256). Same definition as src/utils.py (this script imports nothing from
# src/); src/main.py checks that the summary's copy equals the one it computes from the manifest.
_NORM_FINGERPRINT_KEYS: Tuple[str, ...] = (
    "normalization_methods",
    "per_key_stats",
    "epsilon",
    "min_std",
    "dt",
    "species_variables",
    "global_variables",
)
_REQUIRED_DT_SAMPLING = "loguniform"
_ALLOWED_GLOBAL_METHODS = {"standard", "min-max", "log-min-max", "log-standard"}
# Config keys starting with this prefix are comments (same rule as src/utils.py); any other
# '_'-prefixed key is unknown.
_COMMENT_KEY_PREFIX = "_comment"
# One preprocessing run per processed_dir: <processed_dir>.lock is flocked for the whole run
# (readers lock the same file shared: src/utils.py PROCESSED_DIR_LOCK_SUFFIX must equal this).
_LOCK_SUFFIX = ".lock"
# prctl(2) option from <linux/prctl.h>: signal this process when its parent thread exits.
_PR_SET_PDEATHSIG = 1
_REQUIRED_TOP_LEVEL: Tuple[str, ...] = ("paths", "data", "normalization", "preprocessing", "system")
_ALLOWED_TOP_LEVEL: Tuple[str, ...] = (
    "precision",
    "paths",
    "normalization",
    "preprocessing",
    "system",
    "runtime",
    "data",
    "dataset",
    "model",
    "training",
)
_REQUIRED_SECTION_KEYS: Dict[str, Tuple[str, ...]] = {
    "paths": ("raw_dir", "processed_dir", "work_dir"),
    "data": ("global_variables", "species_variables"),
    "normalization": ("epsilon", "min_std", "methods"),
    "preprocessing": (
        "raw_file_patterns",
        "dt_min",
        "dt_max",
        "dt_sampling",
        "n_steps",
        "t_min",
        "output_trajectories_per_file",
        "shard_size",
        "overwrite",
        "time_key",
        "val_fraction",
        "test_fraction",
        "seed",
        "pool_size",
        "samples_per_source_trajectory",
        "max_chunk_attempts_per_source",
        "drop_below",
        "num_workers",
    ),
    "system": ("device", "log_level", "seed"),
}


def _validate_no_unknown_keys(mapping: Mapping[str, Any], *, allowed: Sequence[str], prefix: str = "") -> None:
    """Reject any key not in `allowed` ('_comment*' keys excepted; whitespace-ambiguous keys fail)."""
    allowed_set = set(allowed)
    for raw_key in mapping.keys():
        if not isinstance(raw_key, str):
            raise TypeError("bad key")
        if raw_key.strip() != raw_key:
            raise KeyError("ambiguous key")
        if raw_key.startswith(_COMMENT_KEY_PREFIX):
            continue
        if raw_key not in allowed_set:
            path = f"{prefix}.{raw_key}" if prefix else raw_key
            raise KeyError(f"unknown: {path}")


def validate_required_config_keys(cfg: Mapping[str, Any]) -> None:
    """Strictly validate the preprocessing-relevant config sections and keys.

    Checks top-level sections, rejects unknown keys, and requires every key the
    pipeline reads (every allowed section key is required). Only the sections
    preprocessing depends on are validated here (training/runtime/etc. are
    validated by src/main.py).
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("bad config")

    _validate_no_unknown_keys(cfg, allowed=_ALLOWED_TOP_LEVEL)
    for sec in _REQUIRED_TOP_LEVEL:
        if sec not in cfg:
            raise KeyError(f"missing: {sec}")

    for sec, required_keys in _REQUIRED_SECTION_KEYS.items():
        block = cfg[sec]
        if not isinstance(block, Mapping):
            raise TypeError(f"bad type: {sec}")

        _validate_no_unknown_keys(block, allowed=required_keys, prefix=sec)
        for key in required_keys:
            if key not in block:
                raise KeyError(f"missing: {sec}.{key}")


# -----------------------------------------------------------------------------
# Strict config parsing
# -----------------------------------------------------------------------------


def _parse_global_method(method: str, *, key: str) -> str:
    """Normalize and validate a global-normalization method name (raise if unsupported)."""
    m = str(method).lower().strip()
    if m not in _ALLOWED_GLOBAL_METHODS:
        raise ValueError(
            f"Unsupported normalization method '{method}' for '{key}'. "
            f"Supported: {sorted(_ALLOWED_GLOBAL_METHODS)}"
        )
    return m


def _require(mapping: Dict, key: str) -> object:
    """Return mapping[key] or raise KeyError naming the missing config key."""
    if key not in mapping:
        raise KeyError(f"missing config key: {key}")
    return mapping[key]


def _require_dict(mapping: Dict, key: str) -> Dict:
    """Return a required nested object (dict), or raise TypeError."""
    obj = _require(mapping, key)
    if not isinstance(obj, dict):
        raise TypeError(f"config.{key} must be an object")
    return obj


def _require_str(mapping: Dict, key: str) -> str:
    """Return a required non-empty string (stripped), or raise TypeError."""
    obj = _require(mapping, key)
    if not isinstance(obj, str) or not obj.strip():
        raise TypeError(f"config.{key} must be a non-empty string")
    return obj.strip()


def _require_int(mapping: Dict, key: str) -> int:
    """Return a required int (bools rejected), or raise TypeError."""
    obj = _require(mapping, key)
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise TypeError(f"config.{key} must be an int")
    return int(obj)


def _require_optional_cap(mapping: Dict, key: str) -> Optional[int]:
    """Return a required cap: null (no cap) or an int > 0 (bools rejected), or raise TypeError/ValueError."""
    obj = _require(mapping, key)
    if obj is None:
        return None
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise TypeError(f"config.{key} must be null (no cap) or an int")
    if obj <= 0:
        raise ValueError(f"config.{key} must be null (no cap) or > 0, got {obj}")
    return int(obj)


def _require_float(mapping: Dict, key: str) -> float:
    """Return a required finite number as float (bools rejected), or raise TypeError/ValueError."""
    obj = _require(mapping, key)
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise TypeError(f"config.{key} must be a number")
    if not math.isfinite(obj):
        raise ValueError(f"config.{key} must be finite, got {obj!r}")
    return float(obj)


def _require_bool(mapping: Dict, key: str) -> bool:
    """Return a required bool, or raise TypeError."""
    obj = _require(mapping, key)
    if not isinstance(obj, bool):
        raise TypeError(f"config.{key} must be a bool")
    return bool(obj)


def _require_str_list(mapping: Dict, key: str) -> List[str]:
    """Return a required non-empty list of non-empty strings (each stripped)."""
    obj = _require(mapping, key)
    if not isinstance(obj, list) or not obj:
        raise TypeError(f"config.{key} must be a non-empty list")
    out: List[str] = []
    for v in obj:
        if not isinstance(v, str) or not v.strip():
            raise TypeError(f"config.{key} must contain non-empty strings")
        out.append(v.strip())
    return out


def _resolve_path(root: Path, p: str) -> Path:
    """Resolve a possibly-relative path against `root` (the config-file directory)."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class PreCfg:
    """Fully parsed, validated preprocessing configuration (paths resolved to absolute)."""

    raw_dir: Path
    processed_dir: Path
    raw_file_patterns: List[str]

    dt_min: float
    dt_max: float
    dt_sampling: str  # loguniform only

    n_steps: int
    t_min: float

    # Per-file caps; None = no cap (consume every group). A finite value is a deliberate subsample:
    # pool_size caps the source groups drawn from a file (reservoir sample), and
    # output_trajectories_per_file caps the windows written per file (samples_per_source_trajectory
    # per kept group), so it binds first. Both are recorded per file in preprocessing_summary.json.
    output_trajectories_per_file: Optional[int]
    shard_size: int
    overwrite: bool

    time_key: str

    val_fraction: float
    test_fraction: float

    global_variables: List[str]
    species_variables: List[str]

    epsilon: float
    min_std: float
    methods: Dict[str, str]

    seed: int
    pool_size: Optional[int]
    samples_per_source_trajectory: int
    max_chunk_attempts_per_source: int
    drop_below: float
    num_workers: int  # number of parallel worker processes for the per-file sampling fan-out


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """json object_pairs_hook: a repeated key is an error (plain json.load keeps the last one)."""
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _reject_non_finite_constant(name: str) -> Any:
    """json parse_constant hook: NaN / Infinity / -Infinity are not valid config values."""
    raise ValueError(f"non-finite JSON number {name} is not allowed")


def load_json(path: Path) -> Dict:
    """Load a JSON config file strictly (no duplicate keys, no NaN/Infinity), requiring a top-level object."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_non_finite_constant)
        except ValueError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(obj, dict):
        raise TypeError("Config must be a JSON object")
    return obj


def parse_precfg(cfg: Dict, *, cfg_path: Path) -> PreCfg:
    """Parse and validate the raw config dict into a PreCfg (strict types + range checks).

    Resolves raw_dir/processed_dir relative to the config-file directory, enforces the
    exact (P, T) global set, the loguniform dt contract, and all positivity constraints.
    """
    root = cfg_path.parent.resolve()

    paths = _require_dict(cfg, "paths")
    raw_dir = _resolve_path(root, _require_str(paths, "raw_dir"))
    processed_dir = _resolve_path(root, _require_str(paths, "processed_dir"))

    data = _require_dict(cfg, "data")
    global_variables = _require_str_list(data, "global_variables")
    species_variables = _require_str_list(data, "species_variables")
    if global_variables != list(_REQUIRED_GLOBALS):
        raise ValueError(f"config.data.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")
    dups = sorted({sp for sp in species_variables if species_variables.count(sp) > 1})
    if dups:
        raise ValueError(f"config.data.species_variables has duplicate names: {dups}")

    norm = _require_dict(cfg, "normalization")
    epsilon = _require_float(norm, "epsilon")
    min_std = _require_float(norm, "min_std")
    methods_obj = _require(norm, "methods")
    if not isinstance(methods_obj, dict):
        raise TypeError("config.normalization.methods must be an object")

    raw_method_keys = [k.strip() for k in methods_obj.keys() if isinstance(k, str)]
    if len(raw_method_keys) != len(methods_obj):
        raise TypeError("config.normalization.methods must map strings to strings")
    if len(set(raw_method_keys)) != len(raw_method_keys):
        raise ValueError("config.normalization.methods contains duplicate keys after whitespace normalization")
    missing_keys = [g for g in global_variables if g not in raw_method_keys]
    unexpected_keys = [k for k in raw_method_keys if k not in global_variables]
    if missing_keys or unexpected_keys:
        raise ValueError(
            "config.normalization.methods must define exactly data.global_variables. "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )

    methods: Dict[str, str] = {}
    for k, v in methods_obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("config.normalization.methods must map strings to strings")
        key = k.strip()
        methods[key] = _parse_global_method(v, key=key)

    pr = _require_dict(cfg, "preprocessing")
    raw_file_patterns = _require_str_list(pr, "raw_file_patterns")

    dt_min = _require_float(pr, "dt_min")
    dt_max = _require_float(pr, "dt_max")
    if dt_min <= 0.0 or dt_max <= 0.0 or dt_max <= dt_min:
        raise ValueError("Invalid dt range: require 0 < dt_min < dt_max")

    dt_sampling = _require_str(pr, "dt_sampling").lower()
    if dt_sampling != _REQUIRED_DT_SAMPLING:
        raise ValueError(f"preprocessing.dt_sampling must be '{_REQUIRED_DT_SAMPLING}'")

    n_steps = _require_int(pr, "n_steps")
    if n_steps < 2:
        raise ValueError("preprocessing.n_steps must be >= 2")

    t_min = _require_float(pr, "t_min")
    if t_min < 0.0:
        raise ValueError("preprocessing.t_min must be >= 0")

    output_trajectories_per_file = _require_optional_cap(pr, "output_trajectories_per_file")
    shard_size = _require_int(pr, "shard_size")
    overwrite = _require_bool(pr, "overwrite")

    time_key = _require_str(pr, "time_key")

    val_fraction = _require_float(pr, "val_fraction")
    test_fraction = _require_float(pr, "test_fraction")
    if val_fraction < 0.0 or test_fraction < 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError("Require 0 <= val_fraction, 0 <= test_fraction, and val_fraction + test_fraction < 1")

    seed = _require_int(pr, "seed")
    pool_size = _require_optional_cap(pr, "pool_size")
    samples_per_source_trajectory = _require_int(pr, "samples_per_source_trajectory")
    max_chunk_attempts_per_source = _require_int(pr, "max_chunk_attempts_per_source")
    drop_below = _require_float(pr, "drop_below")
    num_workers = _require_int(pr, "num_workers")

    if num_workers <= 0:
        raise ValueError("preprocessing.num_workers must be > 0")
    if shard_size <= 0:
        raise ValueError("preprocessing.shard_size must be > 0")
    if samples_per_source_trajectory <= 0:
        raise ValueError("preprocessing.samples_per_source_trajectory must be > 0")
    if max_chunk_attempts_per_source <= 0:
        raise ValueError("preprocessing.max_chunk_attempts_per_source must be > 0")
    if epsilon <= 0.0:
        raise ValueError("normalization.epsilon must be > 0")
    if min_std <= 0.0:
        raise ValueError("normalization.min_std must be > 0")
    # Every kept value is >= drop_below >= epsilon, so the epsilon floors in the log transforms
    # never change a kept value.
    if drop_below < epsilon:
        raise ValueError(
            f"preprocessing.drop_below ({drop_below}) must be >= normalization.epsilon ({epsilon})"
        )

    return PreCfg(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_file_patterns=raw_file_patterns,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_sampling=dt_sampling,
        n_steps=n_steps,
        t_min=t_min,
        output_trajectories_per_file=output_trajectories_per_file,
        shard_size=shard_size,
        overwrite=overwrite,
        time_key=time_key,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        global_variables=global_variables,
        species_variables=species_variables,
        epsilon=epsilon,
        min_std=min_std,
        methods=methods,
        seed=seed,
        pool_size=pool_size,
        samples_per_source_trajectory=samples_per_source_trajectory,
        max_chunk_attempts_per_source=max_chunk_attempts_per_source,
        drop_below=drop_below,
        num_workers=num_workers,
    )


# -----------------------------------------------------------------------------
# File utilities
# -----------------------------------------------------------------------------


def list_raw_files(raw_dir: Path, patterns: Sequence[str]) -> List[Path]:
    """Return the sorted, de-duplicated set of raw files under `raw_dir` matching any pattern."""
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(raw_dir.glob(pat)))
    uniq = sorted({p.resolve() for p in files})
    return uniq


def clean_dir(path: Path, *, overwrite: bool) -> None:
    """Create an empty directory; if it exists, recreate it (overwrite) or refuse."""
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def refuse_existing_outputs(processed_dir: Path) -> None:
    """Raise if processed_dir already holds a split dir or the manifest/summary (overwrite=false)."""
    for name in ("train", "validation", "test", "normalization.json", "preprocessing_summary.json"):
        p = processed_dir / name
        if p.exists():
            raise RuntimeError(f"Refusing to overwrite existing output (preprocessing.overwrite=false): {p}")


def remove_tmp_dir(path: Path) -> None:
    """Delete the temporary physical-shard directory; raise if it is missing or not removed."""
    if not path.exists():
        raise RuntimeError(f"Expected temporary directory to exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"Temporary path is not a directory: {path}")
    shutil.rmtree(path)
    if path.exists():
        raise RuntimeError(f"Failed to remove temporary directory: {path}")


def flush_shard(
    out_dir: Path,
    *,
    split: str,
    shard_idx: int,
    y_buf: List[np.ndarray],
    g_buf: List[np.ndarray],
    dt_buf: List[np.ndarray],
    suffix: str,
    name_prefix: str = "shard",
) -> Tuple[int, int]:
    """Stack the buffered samples and write one NPZ shard; return (next_idx, n_written).

    Always the physical schema (y_mat/globals/dt_mat): both callers pass suffix="_physical", and
    the normalized shards are written by the normalization pass. Empty buffers no-op.

    `name_prefix` namespaces the shard filename so that the parallel per-file workers
    write non-colliding paths (e.g. ``shard_f0007_000003_physical.npz``). Downstream
    ``iter_shards`` globs ``shard_*{suffix}.npz``, so any prefix starting with ``shard``
    is discovered.
    """
    if not y_buf:
        return shard_idx, 0

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    y_mat = np.stack(y_buf, axis=0)
    g_mat = np.stack(g_buf, axis=0)
    dt_mat = np.stack(dt_buf, axis=0)

    shard_path = split_dir / f"{name_prefix}_{shard_idx:06d}{suffix}.npz"

    np.savez(shard_path, y_mat=y_mat, globals=g_mat, dt_mat=dt_mat)

    return shard_idx + 1, int(y_mat.shape[0])


def iter_shards(root: Path, split: str, suffix: str) -> List[Path]:
    """Return the sorted shard paths for a split matching the given suffix (empty if none)."""
    d = root / split
    return sorted(d.glob(f"shard_*{suffix}.npz")) if d.exists() else []


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


def reservoir_sample(keys: Iterable[str], k: int, rng: np.random.Generator) -> List[str]:
    """Uniformly sample up to k items from a stream of keys without materializing it (reservoir)."""
    pool: List[str] = []
    for i, key in enumerate(keys):
        if i < k:
            pool.append(key)
            continue
        j = int(rng.integers(0, i + 1))
        if j < k:
            pool[j] = key
    return pool


def split_for_trajectory(file_name: str, traj_name: str, *, seed: int, val_frac: float, test_frac: float) -> str:
    """Deterministically assign a trajectory to train/validation/test (SPLIT_KEY_DEFINITION, seed=SPLIT_SEED)."""
    key = f"{file_name}:{traj_name}"
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    u = (int.from_bytes(h[:_SPLIT_HASH_BYTES], "big") + 0.5) / 2 ** (8 * _SPLIT_HASH_BYTES)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "validation"
    return "train"


def _seed_word(name: str) -> int:
    """One 64-bit word of SHA-1(name): a stable per-identity seed component and group-name digest."""
    return int.from_bytes(hashlib.sha1(name.encode()).digest()[:_SEED_WORD_BYTES], "big")


def sample_dt(cfg: PreCfg, rng: np.random.Generator) -> float:
    """Draw a single physical dt log-uniformly from [dt_min, dt_max] (seconds)."""
    lo = np.log10(cfg.dt_min)
    hi = np.log10(cfg.dt_max)
    return float(10.0 ** rng.uniform(lo, hi))


def pick_t_start(
    *,
    t_raw: np.ndarray,
    t_valid: np.ndarray,
    dt_s: float,
    n_steps: int,
    t_min: float,
    rng: np.random.Generator,
    anchor_first: bool,
    max_attempts: int,
) -> Optional[float]:
    """Find a chunk start time so the full [n_steps]-point chunk fits inside the valid window.

    The first attempt may anchor at the earliest feasible time (anchor_first); otherwise
    candidates are drawn log-uniformly in [t_lo, t_hi]. Returns None if no placement fits.

    Candidates end at the valid window's last time (t_hi = t_valid[-1] - duration), which is the
    last raw time unless frozen_tail_cut shortened the window. t_lo is at or after t_valid[0], so
    a candidate fits unless float rounding puts it just past t_hi, and a window longer than the
    valid span returns None without a draw.

    Callers pass groups with at least 2 positive times (sample_file and processing/testing.py
    reject the others first; testing/eval_rollout_error.py replays only kept groups), so t_raw
    has a positive entry here.
    """
    pos = t_raw > 0
    first_pos_time = float(t_raw[pos][0])
    t_lo = max(float(t_min), first_pos_time)

    chunk_offsets = np.arange(n_steps, dtype=np.float64) * float(dt_s)
    chunk_duration = float(chunk_offsets[-1])

    t_end = float(t_valid[-1])
    t_hi = t_end - chunk_duration
    if t_hi <= t_lo:
        return None

    def fits(start: float) -> bool:
        t_chunk = start + chunk_offsets
        return (t_chunk[0] >= float(t_valid[0])) and (t_chunk[-1] <= float(t_valid[-1]))

    if anchor_first and fits(t_lo):
        return float(t_lo)

    lo = np.log10(t_lo)
    hi = np.log10(t_hi)

    for _ in range(max_attempts):
        cand = float(10.0 ** rng.uniform(lo, hi))
        if fits(cand):
            return cand

    return None


def frozen_tail_cut(y_raw: np.ndarray, *, epsilon: float) -> Optional[int]:
    """The raw row f at which a trajectory's frozen tail is cut (keep rows <= f), or None.

    f is the first row of the trailing run of rows whose species are all bitwise identical to the
    last row (the generator stopped and repeated its state). The tail is cut only when the state
    froze in mid-evolution: f >= 1 and the raw step into row f, max over species of
    |log10 max(y[f], epsilon) - log10 max(y[f-1], epsilon)|, exceeds _FROZEN_TAIL_MAX_PRECHANGE_DEX.
    Row f, the last raw point at which a species still changed by more than that, is kept; rows
    after it are dropped. None (nothing cut) for a converged tail (the step into f at or below the
    threshold), for a trajectory whose last two rows differ, and for one constant from row 0.
    Run detection is bitwise; a 1e-12 dex tolerance selects the same groups on the 4 raw files.
    """
    same = np.all(y_raw == y_raw[-1], axis=1)
    differ = np.flatnonzero(np.logical_not(same))
    if differ.size == 0:
        return None
    f = int(differ[-1]) + 1
    if f == y_raw.shape[0] - 1:
        return None
    step = np.abs(np.log10(np.maximum(y_raw[f], epsilon)) - np.log10(np.maximum(y_raw[f - 1], epsilon)))
    return f if float(np.max(step)) > _FROZEN_TAIL_MAX_PRECHANGE_DEX else None


# -----------------------------------------------------------------------------
# HDF5 reading (strict)
# -----------------------------------------------------------------------------


def leaf_dataset_index(grp: h5py.Group) -> Dict[str, List[str]]:
    """Map each leaf dataset name to the list of full paths having that leaf (for ambiguity checks)."""
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def _unique_dataset_path(grp: h5py.Group, leaf_index: Dict[str, List[str]], key: str) -> str:
    """Resolve a dataset key (full path or unambiguous leaf) to its full path; raise if missing/ambiguous."""
    if "/" in key:
        if key not in grp:
            raise KeyError(f"Dataset path not found in trajectory group: {key}")
        obj = grp[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Expected dataset at {key}")
        return key

    matches = leaf_index.get(key, [])
    if not matches:
        raise KeyError(f"Dataset not found in trajectory group: {key}")
    if len(matches) != 1:
        raise KeyError(f"Dataset name is ambiguous in trajectory group: {key} -> {matches}")
    return matches[0]


def read_time(grp: h5py.Group, *, time_key: str, leaf_index: Dict[str, List[str]]) -> np.ndarray:
    """Read the 1-D time vector (float64), requiring >=2 finite, strictly increasing samples."""
    p = _unique_dataset_path(grp, leaf_index, time_key)

    t = np.asarray(grp[p][...], dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError(f"time dataset '{time_key}' has < 2 samples")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"time dataset '{time_key}' contains non-finite values")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"time dataset '{time_key}' is not strictly increasing")
    return t


def read_species_matrix(
    grp: h5py.Group,
    *,
    t_len: int,
    species_vars: Sequence[str],
    leaf_index: Dict[str, List[str]],
) -> np.ndarray:
    """Read species into a [t_len, S] float64 matrix; each must be scalar-per-time and finite."""
    s = len(species_vars)
    y = np.empty((t_len, s), dtype=np.float64)

    for j, name in enumerate(species_vars):
        p = _unique_dataset_path(grp, leaf_index, name)
        arr = np.asarray(grp[p][...], dtype=np.float64)

        if arr.ndim == 1:
            if int(arr.shape[0]) != int(t_len):
                raise ValueError(
                    f"Species dataset '{name}' must have shape [{t_len}] or [{t_len}, 1], got {arr.shape}"
                )
            col = arr
        elif arr.ndim == 2 and int(arr.shape[0]) == int(t_len) and int(arr.shape[1]) == 1:
            col = arr[:, 0]
        else:
            raise ValueError(
                f"Species dataset '{name}' must be scalar-per-time with shape [{t_len}] or [{t_len}, 1], got {arr.shape}"
            )

        if not np.all(np.isfinite(col)):
            raise ValueError(f"Species dataset '{name}' contains non-finite values")
        y[:, j] = col

    return y


def read_globals_vector(
    grp: h5py.Group,
    *,
    global_vars: Sequence[str],
    leaf_index: Dict[str, List[str]],
) -> np.ndarray:
    """Read per-trajectory globals as a [G] float32 vector from group attrs or scalar datasets.

    Each global must be present as exactly one of attribute OR dataset (both present is
    ambiguous and rejected), hold exactly one value (else TypeError: a schema mismatch), be
    finite (else the 'must be a finite scalar' ValueError: a non_finite reject) and be > 0 as
    float32 (else the 'must be > 0' ValueError: a non_positive_global reject). P and T are
    physically positive; a tiny positive float64 P that underflows to 0 in the cast fails too.
    """
    out = np.empty((len(global_vars),), dtype=np.float64)

    for j, name in enumerate(global_vars):
        attr_present = name in grp.attrs
        if "/" in name:
            ds_present = name in grp and isinstance(grp[name], h5py.Dataset)
        else:
            ds_present = len(leaf_index.get(name, [])) > 0

        if attr_present and ds_present:
            raise ValueError(
                f"Global '{name}' is ambiguous: both attribute and dataset are present in trajectory group."
            )

        if attr_present:
            v = np.asarray(grp.attrs[name], dtype=np.float64).reshape(-1)
            if v.size != 1:
                raise TypeError(f"Global attribute '{name}' must be a scalar, got {v.size} values")
            if not np.isfinite(v[0]):
                raise ValueError(f"Global attribute '{name}' must be a finite scalar, got {v[0]!r}")
            if not np.float32(v[0]) > 0.0:
                raise ValueError(f"Global '{name}' must be > 0 (as float32), got {v[0]!r}")
            out[j] = float(v[0])
            continue

        # Otherwise require a scalar dataset in the trajectory group.
        p = _unique_dataset_path(grp, leaf_index, name)
        ds = grp[p]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"Global '{name}' is not a dataset")

        arr = np.asarray(ds[...], dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise TypeError(f"Global '{name}' must be a scalar dataset; got shape {ds.shape}")
        if not np.isfinite(arr[0]):
            raise ValueError(f"Global '{name}' must be a finite scalar dataset, got {arr[0]!r}")
        if not np.float32(arr[0]) > 0.0:
            raise ValueError(f"Global '{name}' must be > 0 (as float32), got {arr[0]!r}")
        out[j] = float(arr[0])

    return out.astype(np.float32)


def _is_non_finite_raw_error(err: BaseException) -> bool:
    """True if `err` reports a non-finite raw value (a soft reject, not a schema error)."""
    msg = str(err)
    return "contains non-finite values" in msg or "must be a finite scalar" in msg


def _is_non_positive_global_error(err: BaseException) -> bool:
    """True if `err` reports a P/T value that is not > 0 as float32 (a soft reject)."""
    return "must be > 0 (as float32)" in str(err)


# -----------------------------------------------------------------------------
# Interpolation
# -----------------------------------------------------------------------------


def prepare_log_interp(log_t_valid: np.ndarray, log_t_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute (i0, i1, weight) bracketing indices for linear interpolation on the log-time grid."""
    n = int(log_t_valid.shape[0])
    idx = np.searchsorted(log_t_valid, log_t_target, side="left")
    idx = np.clip(idx, 1, n - 1)

    i0 = (idx - 1).astype(np.int64)
    i1 = idx.astype(np.int64)

    t0 = log_t_valid[i0]
    t1 = log_t_valid[i1]
    denom = t1 - t0
    if np.any(denom <= 0):
        raise ValueError("Invalid time grid: encountered non-positive interpolation denominator")

    w = (log_t_target - t0) / denom
    w = np.clip(w, 0.0, 1.0).astype(np.float64)
    return i0, i1, w


def interp_loglog(
    y_valid: np.ndarray,
    *,
    i0: np.ndarray,
    i1: np.ndarray,
    w: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Interpolate species in log-log space: log10(y) is linear in log10(t), then exponentiated.

    Uses the bracketing indices/weights from prepare_log_interp; values are floored at epsilon
    before log10 to avoid -inf.
    """
    y0 = np.log10(np.maximum(y_valid[i0, :], epsilon))
    y1 = np.log10(np.maximum(y_valid[i1, :], epsilon))
    w2 = w.reshape(-1, 1)
    return 10.0 ** ((1.0 - w2) * y0 + w2 * y1)


# -----------------------------------------------------------------------------
# Core preprocessing
# -----------------------------------------------------------------------------


def _coverage_cell(g_vec: np.ndarray) -> Tuple[int, int]:
    """(T bin, log10 P bin) indices of a group, floor-based (P > 0 is guaranteed by read_globals_vector)."""
    p_val, t_val = float(g_vec[0]), float(g_vec[1])
    return (
        int(math.floor(t_val / _COVERAGE_T_BIN_K)),
        int(math.floor(math.log10(p_val) / _COVERAGE_LOG10P_BIN_DEX)),
    )


def _new_cell() -> Dict[str, Any]:
    return {"too_few_valid": 0, "reached_drop_test": 0, "drop_below": 0, "frozen_tail_no_window": 0,
            "windows": {sp: 0 for sp in _SPLITS}}


def _hist_add(hist: Dict[int, int], value: float, width: float) -> None:
    """Count log10(value) in the floor-based bin of `width` dex (keyed by the bin index)."""
    b = int(math.floor(math.log10(value) / width))
    hist[b] = hist.get(b, 0) + 1


def sample_file(
    file_path: Path,
    file_idx: int,
    *,
    out_tmp: Path,
    cfg: PreCfg,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample and resample trajectories from a single HDF5 file into physical shards.

    Self-contained so it can run in a worker process. Generators: one per file, seeded from
    ``cfg.seed`` and the file name, only reservoir-samples and shuffles the group pool; one per
    trajectory, seeded from ``cfg.seed`` and the group name, draws that trajectory's windows (dt and
    start time), so a window does not depend on the file's position in the list or on which other
    groups were kept. ``file_idx`` only namespaces this file's shard names.

    A kept group whose raw tail froze in mid-evolution (frozen_tail_cut) is sampled only up to the
    cut; if none of its draws fits there it is rejected as frozen_tail_no_window instead of kept.

    Returns ``(row, agg)``. ``row`` is this file's preprocessing_summary.json entry: name, size,
    group counts (groups_seen = sum(group_rejects) + groups_kept), group- and chunk-level rejects,
    windows written per split, ``frozen_tail`` (per split: groups cut, their draws that found no
    placement, counted in no_fit_chunk too, and cut groups left with no window), ``truncated``
    (the per-file output cap stopped sampling while candidate draws remained), shards per split
    and the raw time grid of the first group with a finite time vector. ``agg`` holds the counters
    the parent sums over files: the (T, log10 P) coverage cells, drop_below trigger counts, window
    start-time / dt histograms, anchored windows, per-split window start-time range, and the 64-bit
    digests of every group name in the file (the cross-file duplicate check).
    """
    rng = np.random.default_rng([int(cfg.seed), _seed_word(file_path.name)])
    shard_idx: Dict[str, int] = {sp: 0 for sp in _SPLITS}
    name_prefix = f"shard_f{int(file_idx):04d}"

    y_buf: Dict[str, List[np.ndarray]] = {sp: [] for sp in _SPLITS}
    g_buf: Dict[str, List[np.ndarray]] = {sp: [] for sp in _SPLITS}
    dt_buf: Dict[str, List[np.ndarray]] = {sp: [] for sp in _SPLITS}

    # Group-level rejects (a whole source group) and chunk-level rejects (one window attempt).
    group_rejects: Dict[str, int] = {
        "non_finite": 0, "non_positive_global": 0, "too_few_valid": 0, "drop_below": 0, "frozen_tail_no_window": 0,
    }
    chunk_rejects: Dict[str, int] = {"no_fit_chunk": 0, "interp_non_finite": 0}
    frozen_tail: Dict[str, Dict[str, int]] = {
        k: {sp: 0 for sp in _SPLITS} for k in ("truncated_groups", "removed_windows", "lost_groups")
    }
    windows_by_split = {sp: 0 for sp in _SPLITS}
    groups_seen = 0
    groups_kept = 0
    raw_time_grid: Optional[Dict[str, Any]] = None

    coverage: Dict[Tuple[int, int], Dict[str, Any]] = {}
    drop_triggers = {sp: 0 for sp in cfg.species_variables}
    drop_exact_zero = 0
    t_start_hist: Dict[str, Dict[int, int]] = {sp: {} for sp in _SPLITS}
    dt_hist: Dict[str, Dict[int, int]] = {sp: {} for sp in _SPLITS}
    anchored = {sp: 0 for sp in _SPLITS}
    t_start_range: Dict[str, List[float]] = {}

    target = cfg.output_trajectories_per_file  # None: no cap
    written = 0
    truncated = False

    step_offsets = np.arange(cfg.n_steps, dtype=np.float64)

    with h5py.File(file_path, "r") as fin:
        n_groups = len(fin)
        name_digests = np.fromiter((_seed_word(str(k)) for k in fin.keys()), dtype=np.uint64, count=n_groups)
        # No cap: every group. The shuffle stays unconditional (reservoir_sample with k >= n keeps
        # the key order and draws nothing, so both branches give the same pool and RNG stream).
        pool = list(fin.keys()) if cfg.pool_size is None else reservoir_sample(fin.keys(), cfg.pool_size, rng)
        rng.shuffle(pool)

        for traj_name in pool:
            if target is not None and written >= target:
                truncated = True
                break
            groups_seen += 1

            grp_obj = fin[traj_name]
            if not isinstance(grp_obj, h5py.Group):
                raise TypeError(
                    f"Raw schema mismatch: expected top-level group, got {type(grp_obj).__name__} at '{traj_name}' in {file_path}"
                )

            grp: h5py.Group = grp_obj

            leaf_index = leaf_dataset_index(grp)

            # Time
            try:
                t_raw = read_time(grp, time_key=cfg.time_key, leaf_index=leaf_index)
            except (KeyError, ValueError, TypeError) as e:
                if isinstance(e, ValueError) and _is_non_finite_raw_error(e):
                    group_rejects["non_finite"] += 1
                    log.debug(
                        "Reject trajectory '%s' in %s: non-finite time values (%s)",
                        traj_name,
                        file_path.name,
                        e,
                    )
                    continue
                raise RuntimeError(
                    f"Raw schema mismatch while reading time for trajectory '{traj_name}' in {file_path.name}: {e}"
                ) from e

            T = int(t_raw.shape[0])
            if raw_time_grid is None:
                pos_t = t_raw[t_raw > 0]
                raw_time_grid = {
                    "group": str(traj_name),
                    "n_points": T,
                    "t_first": float(t_raw[0]),
                    "t_first_positive": float(pos_t[0]) if pos_t.size else None,
                    "t_last": float(t_raw[-1]),
                }

            # Species
            try:
                y_raw = read_species_matrix(grp, t_len=T, species_vars=cfg.species_variables, leaf_index=leaf_index)
            except (KeyError, ValueError, TypeError) as e:
                if isinstance(e, ValueError) and _is_non_finite_raw_error(e):
                    group_rejects["non_finite"] += 1
                    log.debug(
                        "Reject trajectory '%s' in %s: non-finite species values (%s)",
                        traj_name,
                        file_path.name,
                        e,
                    )
                    continue
                raise RuntimeError(
                    f"Raw schema mismatch while reading species for trajectory '{traj_name}' in {file_path.name}: {e}"
                ) from e

            # Globals
            try:
                g_vec = read_globals_vector(grp, global_vars=cfg.global_variables, leaf_index=leaf_index)
            except (KeyError, ValueError, TypeError) as e:
                if isinstance(e, ValueError) and _is_non_finite_raw_error(e):
                    group_rejects["non_finite"] += 1
                    log.debug(
                        "Reject trajectory '%s' in %s: non-finite global values (%s)",
                        traj_name,
                        file_path.name,
                        e,
                    )
                    continue
                if isinstance(e, ValueError) and _is_non_positive_global_error(e):
                    group_rejects["non_positive_global"] += 1
                    log.debug(
                        "Reject trajectory '%s' in %s: non-positive global value (%s)",
                        traj_name,
                        file_path.name,
                        e,
                    )
                    continue
                raise RuntimeError(
                    f"Raw schema mismatch while reading globals for trajectory '{traj_name}' in {file_path.name}: {e}"
                ) from e
            g_vec32 = g_vec.astype(np.float32, copy=False)
            cell = coverage.setdefault(_coverage_cell(g_vec32), _new_cell())

            # Validity mask (matches training assumptions):
            # - Only consider positive times (log-time interpolation).
            # - Require at least 2 valid samples.
            valid = t_raw > 0
            if int(np.count_nonzero(valid)) < 2:
                group_rejects["too_few_valid"] += 1
                cell["too_few_valid"] += 1
                continue

            cell["reached_drop_test"] += 1
            below = y_raw < float(cfg.drop_below)
            if np.any(below):
                group_rejects["drop_below"] += 1
                cell["drop_below"] += 1
                for j in np.flatnonzero(np.any(below, axis=0)):
                    drop_triggers[cfg.species_variables[int(j)]] += 1
                if np.any(y_raw == 0.0):
                    drop_exact_zero += 1
                continue

            # A raw tail that froze in mid-evolution is dropped: windows come only from rows <= cut.
            cut = frozen_tail_cut(y_raw, epsilon=cfg.epsilon)
            if cut is not None:
                valid &= np.arange(T) <= cut

            t_valid = t_raw[valid]
            log_t_valid = np.log10(t_valid)
            y_valid = y_raw[valid, :]

            split = split_for_trajectory(
                file_path.name,
                str(traj_name),
                seed=SPLIT_SEED,
                val_frac=cfg.val_fraction,
                test_frac=cfg.test_fraction,
            )
            # This trajectory's own generator: its windows depend only on (seed, group name).
            traj_rng = np.random.default_rng([int(cfg.seed), _seed_word(str(traj_name))])
            # The anchor of draw 0 (pick_t_start's t_lo).
            t_anchor = max(float(cfg.t_min), float(t_valid[0]))
            written_before = written

            for sidx in range(int(cfg.samples_per_source_trajectory)):
                if target is not None and written >= target:
                    truncated = True
                    break

                dt_s = sample_dt(cfg, traj_rng)
                t_start = pick_t_start(
                    t_raw=t_raw,
                    t_valid=t_valid,
                    dt_s=dt_s,
                    n_steps=cfg.n_steps,
                    t_min=cfg.t_min,
                    rng=traj_rng,
                    anchor_first=(sidx == 0),
                    max_attempts=cfg.max_chunk_attempts_per_source,
                )
                if t_start is None:
                    chunk_rejects["no_fit_chunk"] += 1
                    if cut is not None:
                        frozen_tail["removed_windows"][split] += 1
                    continue

                t_chunk = float(t_start) + step_offsets * float(dt_s)
                log_t_chunk = np.log10(t_chunk)

                i0, i1, w = prepare_log_interp(log_t_valid, log_t_chunk)
                y_new = interp_loglog(y_valid, i0=i0, i1=i1, w=w, epsilon=cfg.epsilon)

                if not np.all(np.isfinite(y_new)):
                    chunk_rejects["interp_non_finite"] += 1
                    continue

                y_buf[split].append(y_new.astype(np.float32, copy=False))
                g_buf[split].append(g_vec32)
                dt_buf[split].append(np.full(cfg.n_steps - 1, dt_s, dtype=np.float32))

                windows_by_split[split] += 1
                written += 1
                cell["windows"][split] += 1
                _hist_add(t_start_hist[split], float(t_start), _TSTART_HIST_BIN_DEX)
                _hist_add(dt_hist[split], float(dt_s), _DT_HIST_BIN_DEX)
                if sidx == 0 and float(t_start) == t_anchor:
                    anchored[split] += 1
                lo_hi = t_start_range.setdefault(split, [float(t_start), float(t_start)])
                lo_hi[0] = min(lo_hi[0], float(t_start))
                lo_hi[1] = max(lo_hi[1], float(t_start))

                if len(y_buf[split]) >= cfg.shard_size:
                    shard_idx[split], _ = flush_shard(
                        out_tmp,
                        split=split,
                        shard_idx=shard_idx[split],
                        y_buf=y_buf[split],
                        g_buf=g_buf[split],
                        dt_buf=dt_buf[split],
                        suffix="_physical",
                        name_prefix=name_prefix,
                    )
                    y_buf[split].clear()
                    g_buf[split].clear()
                    dt_buf[split].clear()

            if cut is not None:
                frozen_tail["truncated_groups"][split] += 1
                # No window written means every draw was tried (the output cap is reached only by writing).
                if written == written_before:
                    frozen_tail["lost_groups"][split] += 1
                    group_rejects["frozen_tail_no_window"] += 1
                    cell["frozen_tail_no_window"] += 1
                    continue
            groups_kept += 1

        for sp in _SPLITS:
            shard_idx[sp], _ = flush_shard(
                out_tmp,
                split=sp,
                shard_idx=shard_idx[sp],
                y_buf=y_buf[sp],
                g_buf=g_buf[sp],
                dt_buf=dt_buf[sp],
                suffix="_physical",
                name_prefix=name_prefix,
            )

    log.info(
        "file=%s groups seen=%d kept=%d group_rejects=%s chunk_rejects=%s frozen_tail=%s windows=%d "
        "(train=%d val=%d test=%d) truncated=%s",
        file_path.name,
        groups_seen,
        groups_kept,
        group_rejects,
        chunk_rejects,
        frozen_tail,
        written,
        windows_by_split["train"],
        windows_by_split["validation"],
        windows_by_split["test"],
        truncated,
    )

    row = {
        "file_idx": int(file_idx),
        "name": file_path.name,
        "size_bytes": int(file_path.stat().st_size),
        "n_groups": int(n_groups),
        "n_pooled": len(pool),
        "groups_seen": groups_seen,
        "groups_kept": groups_kept,
        "group_rejects": group_rejects,
        "chunk_rejects": chunk_rejects,
        "frozen_tail": frozen_tail,
        "windows_written_by_split": windows_by_split,
        "truncated": truncated,
        "shards_by_split": dict(shard_idx),
        "raw_time_grid": raw_time_grid,
    }
    agg = {
        "coverage": coverage,
        "drop_triggers": drop_triggers,
        "drop_exact_zero": drop_exact_zero,
        "t_start_hist": t_start_hist,
        "dt_hist": dt_hist,
        "anchored": anchored,
        "t_start_range": t_start_range,
        "name_digests": name_digests,
    }
    return row, agg


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


class RunningMeanVar:
    """Streaming per-dimension mean/variance (Welford merge)."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Fold a batch of rows into the running mean/M2 via the parallel Welford merge."""
        x2 = x.reshape(-1, self.dim).astype(np.float64, copy=False)
        m = int(x2.shape[0])  # >= 1: flush_shard never writes an empty shard

        b_mean = np.mean(x2, axis=0)
        b_M2 = np.sum((x2 - b_mean) ** 2, axis=0)

        if self.n == 0:
            self.n = m
            self.mean = b_mean
            self.M2 = b_M2
            return

        delta = b_mean - self.mean
        n_new = self.n + m
        self.mean = self.mean + delta * (m / n_new)
        self.M2 = self.M2 + b_M2 + (delta**2) * (self.n * m / n_new)
        self.n = n_new

    def finalize(self, *, min_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, std, floored): std floored at min_std, floored = var < min_std**2.

        Raises if fewer than 2 samples were seen. An exactly or nearly constant column is NOT caught
        by `floored`: float roundoff keeps its variance above min_std**2 (log_min == log_max in
        per_key_stats is the trace of such a column).
        """
        if self.n <= 1:
            raise RuntimeError("Insufficient samples to compute variance")
        var = self.M2 / (self.n - 1)
        floor_var = float(min_std) ** 2
        std = np.sqrt(np.maximum(var, floor_var))
        return self.mean, std, var < floor_var


def compute_train_stats(out_tmp: Path, cfg: PreCfg) -> Tuple[Dict, Dict[str, str], Dict[str, List[str]], Dict[str, Any]]:
    """Compute normalization statistics from physical train shards.

    Returns (per_key_stats, methods, std_floored, species_sum): std_floored names every
    (key, field) whose variance was below min_std**2 (std floored at min_std; logged as a warning);
    species_sum is the distribution of the tracked-species sum over every train-split state.
    """

    shards = iter_shards(out_tmp, "train", "_physical")
    if not shards:
        raise RuntimeError("No physical train shards found to compute normalization")

    S = len(cfg.species_variables)
    G = len(cfg.global_variables)

    rms_logy = RunningMeanVar(S)
    logy_min = np.full(S, np.inf, dtype=np.float64)
    logy_max = np.full(S, -np.inf, dtype=np.float64)

    rms_g = RunningMeanVar(G)
    rms_glog = RunningMeanVar(G)
    g_min = np.full(G, np.inf, dtype=np.float64)
    g_max = np.full(G, -np.inf, dtype=np.float64)
    glog_min = np.full(G, np.inf, dtype=np.float64)
    glog_max = np.full(G, -np.inf, dtype=np.float64)

    sum_counts = np.zeros(_SPECIES_SUM_HIST_BINS, dtype=np.int64)
    sum_min, sum_max = np.inf, -np.inf

    for p in tqdm(shards, desc="Computing stats"):
        with np.load(p) as z:
            y = np.asarray(z["y_mat"], dtype=np.float64)
            g = np.asarray(z["globals"], dtype=np.float64)

        ysum = y.sum(axis=2).reshape(-1)
        sum_min, sum_max = min(sum_min, float(ysum.min())), max(sum_max, float(ysum.max()))
        bins = np.clip((ysum * _SPECIES_SUM_HIST_BINS).astype(np.int64), 0, _SPECIES_SUM_HIST_BINS - 1)
        sum_counts += np.bincount(bins, minlength=_SPECIES_SUM_HIST_BINS)

        ylog = np.log10(np.maximum(y, cfg.epsilon))
        y2 = ylog.reshape(-1, S)
        rms_logy.update(y2)
        logy_min = np.minimum(logy_min, np.min(y2, axis=0))
        logy_max = np.maximum(logy_max, np.max(y2, axis=0))

        g2 = g.reshape(-1, G)
        rms_g.update(g2)
        g_min = np.minimum(g_min, np.min(g2, axis=0))
        g_max = np.maximum(g_max, np.max(g2, axis=0))

        glog = np.log10(np.maximum(g2, cfg.epsilon))
        rms_glog.update(glog)
        glog_min = np.minimum(glog_min, np.min(glog, axis=0))
        glog_max = np.maximum(glog_max, np.max(glog, axis=0))

    mu_s, sd_s, fl_s = rms_logy.finalize(min_std=cfg.min_std)
    mu_g, sd_g, fl_g = rms_g.finalize(min_std=cfg.min_std)
    mu_glog, sd_glog, fl_glog = rms_glog.finalize(min_std=cfg.min_std)

    std_floored: Dict[str, List[str]] = {}
    for i, name in enumerate(cfg.species_variables):
        if fl_s[i]:
            std_floored[name] = ["log_std"]
    for i, name in enumerate(cfg.global_variables):
        fields = [f for f, fl in (("std", fl_g[i]), ("log_std", fl_glog[i])) if fl]
        if fields:
            std_floored[name] = fields
    for name, fields in std_floored.items():
        log.warning("normalization: %s %s below min_std=%g; floored at min_std", name, fields, cfg.min_std)

    n_states = int(sum_counts.sum())
    cdf = np.cumsum(sum_counts)
    species_sum = {
        "definition": (
            "sum of the tracked species' mixing ratios per state, over every row of every train-split "
            "window (float32 physical shards, summed in float64)"
        ),
        "n_states": n_states,
        "min": sum_min,
        "max": sum_max,
        "quantiles": {
            str(q): float(int(np.searchsorted(cdf, q * n_states, side="left")) + 1) / _SPECIES_SUM_HIST_BINS
            for q in _SPECIES_SUM_QUANTILE_LEVELS
        },
        "quantile_method": (
            f"upper edge of the first bin of a {_SPECIES_SUM_HIST_BINS}-bin histogram on [0, 1] (sums above 1 "
            "in the last bin) at which the cumulative count reaches level * n_states"
        ),
    }

    per_key_stats: Dict[str, Dict] = {}
    for i, name in enumerate(cfg.species_variables):
        per_key_stats[name] = {
            "log_mean": float(mu_s[i]),
            "log_std": float(sd_s[i]),
            "log_min": float(logy_min[i]),
            "log_max": float(logy_max[i]),
            "epsilon": float(cfg.epsilon),
        }

    for i, name in enumerate(cfg.global_variables):
        per_key_stats[name] = {
            "mean": float(mu_g[i]),
            "std": float(sd_g[i]),
            "min": float(g_min[i]),
            "max": float(g_max[i]),
            "log_mean": float(mu_glog[i]),
            "log_std": float(sd_glog[i]),
            "log_min": float(glog_min[i]),
            "log_max": float(glog_max[i]),
        }

    methods: Dict[str, str] = {}
    for s in cfg.species_variables:
        methods[s] = "log-standard"
    for g in cfg.global_variables:
        methods[g] = cfg.methods[g]

    return per_key_stats, methods, std_floored, species_sum


def normalization_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON (sorted keys, compact) of the manifest's _NORM_FINGERPRINT_KEYS."""
    sub = {k: manifest[k] for k in _NORM_FINGERPRINT_KEYS}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def normalize_and_write(
    *,
    out_tmp: Path,
    out_final: Path,
    cfg: PreCfg,
    per_key_stats: Dict,
    methods: Dict[str, str],
    counts_total: Dict[str, int],
    descriptive: Mapping[str, Any],
) -> str:
    """Normalize every physical shard to z-space and write the final shards + normalization.json.

    Returns the manifest's normalization fingerprint (norm_sha256).

    `descriptive` holds the additive top-level manifest keys that describe the trained domain
    (window_start_time_s, coverage, species_sum, std_floored); units come from UNITS.

    Species use log-standard normalization; globals use their per-key method; dt is log10 +
    min-max clipped to [0, 1]. Each physical shard is deleted once its final shard is written,
    so the peak disk is about one dataset plus one shard. The rows written per split must equal
    the sampling totals (counts_total) before the normalization manifest consumed by
    training/inference is written.
    """
    eps = float(cfg.epsilon)
    S = len(cfg.species_variables)
    G = len(cfg.global_variables)

    out_final.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (out_final / split).mkdir(parents=True, exist_ok=True)

    mu_s = np.array([per_key_stats[s]["log_mean"] for s in cfg.species_variables], dtype=np.float64).reshape(1, 1, S)
    sd_s = np.array([per_key_stats[s]["log_std"] for s in cfg.species_variables], dtype=np.float64).reshape(1, 1, S)

    # Globals normalization: per-global method names (selects the formula below) and vectors.
    g_methods = np.array([methods[name] for name in cfg.global_variables])
    g_mean = np.zeros(G, dtype=np.float64)
    g_std = np.ones(G, dtype=np.float64)
    g_min = np.zeros(G, dtype=np.float64)
    g_rng = np.ones(G, dtype=np.float64)
    glog_mean = np.zeros(G, dtype=np.float64)
    glog_std = np.ones(G, dtype=np.float64)
    glog_min = np.zeros(G, dtype=np.float64)
    glog_rng = np.ones(G, dtype=np.float64)

    for j, name in enumerate(cfg.global_variables):
        m = methods[name]
        st = per_key_stats[name]

        if m == "standard":
            g_mean[j] = float(st["mean"])
            g_std[j] = float(st["std"])
            if g_std[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive std")
            continue

        if m == "min-max":
            g_min[j] = float(st["min"])
            g_rng[j] = float(st["max"]) - g_min[j]
            if g_rng[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive range for min-max")
            continue

        if m == "log-min-max":
            glog_min[j] = float(st["log_min"])
            glog_rng[j] = float(st["log_max"]) - glog_min[j]
            if glog_rng[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive range for log-min-max")
            continue

        if m == "log-standard":
            glog_mean[j] = float(st["log_mean"])
            glog_std[j] = float(st["log_std"])
            if glog_std[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive log-std for log-standard")
            continue

        raise ValueError(f"Unsupported global normalization method '{m}' for '{name}'")

    # dt normalization: log10 + min-max to [0,1]
    dt_log_min = float(np.log10(cfg.dt_min))
    dt_log_max = float(np.log10(cfg.dt_max))
    dt_log_rng = dt_log_max - dt_log_min
    if dt_log_rng <= 0.0:
        raise ValueError("Invalid dt normalization range: dt_max must be greater than dt_min")

    rows_written = {"train": 0, "validation": 0, "test": 0}
    for split in ("train", "validation", "test"):
        physical_shards = iter_shards(out_tmp, split, "_physical")
        for i, p in enumerate(tqdm(physical_shards, desc=f"Normalizing {split}")):
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float64)
                g2 = np.asarray(z["globals"], dtype=np.float64)
                dt = np.asarray(z["dt_mat"], dtype=np.float64)

            y_z = ((np.log10(np.maximum(y, eps)) - mu_s) / sd_s).astype(np.float32)

            g_out = g2

            # standard
            std_mask = g_methods == "standard"
            if np.any(std_mask):
                g_out = np.where(std_mask.reshape(1, G), (g2 - g_mean.reshape(1, G)) / g_std.reshape(1, G), g_out)

            # min-max
            mm_mask = g_methods == "min-max"
            if np.any(mm_mask):
                g_out = np.where(mm_mask.reshape(1, G), (g2 - g_min.reshape(1, G)) / g_rng.reshape(1, G), g_out)

            # log-min-max
            lmm_mask = g_methods == "log-min-max"
            if np.any(lmm_mask):
                g_out = np.where(
                    lmm_mask.reshape(1, G),
                    (np.log10(np.maximum(g2, eps)) - glog_min.reshape(1, G)) / glog_rng.reshape(1, G),
                    g_out,
                )

            # log-standard
            ls_mask = g_methods == "log-standard"
            if np.any(ls_mask):
                g_out = np.where(
                    ls_mask.reshape(1, G),
                    (np.log10(np.maximum(g2, eps)) - glog_mean.reshape(1, G)) / glog_std.reshape(1, G),
                    g_out,
                )

            g_z = g_out.astype(np.float32, copy=False)

            dt_norm = (np.log10(np.maximum(dt, eps)) - dt_log_min) / dt_log_rng
            dt_norm = np.clip(dt_norm, 0.0, 1.0).astype(np.float32)

            np.savez(out_final / split / f"shard_{i:06d}.npz", y_mat=y_z, globals=g_z, dt_norm_mat=dt_norm)
            # After the write, so a failed write keeps its source shard.
            p.unlink()
            rows_written[split] += int(y_z.shape[0])

    # A stray or missing physical shard (e.g. from a concurrent run) must not be published.
    if rows_written != counts_total:
        raise RuntimeError(
            f"normalized row counts {rows_written} differ from the sampled counts {counts_total}; "
            "not writing normalization.json"
        )

    manifest = {
        "normalization_methods": methods,
        "per_key_stats": per_key_stats,
        "epsilon": float(cfg.epsilon),
        "min_std": float(cfg.min_std),
        "dt": {"log_min": dt_log_min, "log_max": dt_log_max},
        "species_variables": list(cfg.species_variables),
        "global_variables": list(cfg.global_variables),
        "units": UNITS,
        **descriptive,
    }

    with open(out_final / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return normalization_sha256(manifest)


def _git_value(args: List[str]) -> Optional[str]:
    """stdout of a read-only git command in this file's checkout, or None (logged) on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            check=True, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("git provenance unavailable (%s): %s", " ".join(args), e)
        return None
    return out.strip()


def _run_record() -> Dict[str, Any]:
    """The volatile part of the summary (differs on every run; determinism checks drop it)."""
    status = _git_value(["--no-optional-locks", "status", "--porcelain", "--untracked-files=no"])
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
    }


def _strip_comments(obj: Any) -> Any:
    """A JSON-like value with every '_comment*' key removed (at any depth)."""
    if isinstance(obj, Mapping):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith(_COMMENT_KEY_PREFIX)}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _hist_rows(hist: Mapping[int, int], width: float) -> List[Dict[str, Any]]:
    return [{"log10_lo": b * width, "count": n} for b, n in sorted(hist.items())]


def write_summary(
    dest_dir: Path,
    *,
    processed_dir: Path,
    cfg: PreCfg,
    cfg_dict: Mapping[str, Any],
    rows: List[Dict[str, Any]],
    totals: Mapping[str, Any],
    norm_sha256: str,
) -> None:
    """Write dest_dir/preprocessing_summary.json: the record of this build.

    norm_sha256 is the normalization fingerprint of the normalization.json written with it.

    Keys: windows_written_by_split, groups_seen / groups_kept, group_rejects (whole source groups:
    non_finite, non_positive_global, too_few_valid, drop_below, frozen_tail_no_window) and
    chunk_rejects (window attempts: no_fit_chunk, interp_non_finite), files_truncated; frozen_tail
    (the frozen_tail_cut threshold and rule, and per split the groups cut, their draws that found
    no placement and the cut groups left with no window; src/main.py refuses a build without it);
    raw_files (one row per file, by file_idx); split (seed, fractions, key definition, raw
    basenames); coverage ((T, log10 P) cells: groups that reached the drop_below test, drop_below
    and frozen_tail_no_window rejects, too_few_valid, windows per split; plus drop_below trigger
    counts); windows (log10 start-time and dt histograms per split, anchored
    windows, start-time range); config (the parsed PreCfg, paths relative to processed_dir) and
    config_blocks (the raw preprocessing / normalization / data blocks, comment keys removed, which
    src/main.py compares with the training config); run (time, git, code hash, versions: volatile).
    """
    base = processed_dir.resolve()
    cfg_serialized: Dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            cfg_serialized[k] = os.path.relpath(str(v.resolve()), str(base))
        else:
            cfg_serialized[k] = v

    cov = totals["coverage"]
    summary = {
        "norm_sha256": norm_sha256,
        "windows_written_by_split": totals["windows_written_by_split"],
        "groups_seen": totals["groups_seen"],
        "groups_kept": totals["groups_kept"],
        "group_rejects": totals["group_rejects"],
        "chunk_rejects": totals["chunk_rejects"],
        "files_truncated": sum(1 for r in rows if r["truncated"]),
        "frozen_tail": {
            "max_prechange_dex": _FROZEN_TAIL_MAX_PRECHANGE_DEX,
            "rule": (
                "f = first row of the trailing run of raw rows whose species are all bitwise identical to "
                "the last row; if f >= 1 and max over species |log10 y[f] - log10 y[f-1]| (floored at "
                "epsilon) > max_prechange_dex, only rows <= f are valid (rows after f are dropped)"
            ),
            "truncated_groups": totals["frozen_tail"]["truncated_groups"],
            "removed_windows": totals["frozen_tail"]["removed_windows"],
            "lost_groups": totals["frozen_tail"]["lost_groups"],
        },
        "raw_files": rows,
        "split": {
            "split_seed": SPLIT_SEED,
            "val_fraction": cfg.val_fraction,
            "test_fraction": cfg.test_fraction,
            "key": SPLIT_KEY_DEFINITION,
            "raw_basenames": sorted(r["name"] for r in rows),
        },
        "coverage": {
            "units": {"T": UNITS["T"], "P": UNITS["P"]},
            "T_bin_K": _COVERAGE_T_BIN_K,
            "log10P_bin_dex": _COVERAGE_LOG10P_BIN_DEX,
            "cells": [
                {"T_lo_K": ti * _COVERAGE_T_BIN_K, "log10P_lo": pi * _COVERAGE_LOG10P_BIN_DEX, **c}
                for (ti, pi), c in sorted(cov.items())
            ],
            "drop_below_trigger_species": totals["drop_triggers"],
            "drop_below_with_exact_zero": totals["drop_exact_zero"],
        },
        "windows": {
            "t_start_s_log10_bin_dex": _TSTART_HIST_BIN_DEX,
            "dt_s_log10_bin_dex": _DT_HIST_BIN_DEX,
            "t_start_hist": {sp: _hist_rows(totals["t_start_hist"][sp], _TSTART_HIST_BIN_DEX) for sp in _SPLITS},
            "dt_hist": {sp: _hist_rows(totals["dt_hist"][sp], _DT_HIST_BIN_DEX) for sp in _SPLITS},
            "anchored_at_first_positive_time": totals["anchored"],
            "t_start_range_s": totals["t_start_range"],
        },
        "config": cfg_serialized,
        "config_blocks": {k: _strip_comments(cfg_dict[k]) for k in ("preprocessing", "normalization", "data")},
        "run": _run_record(),
    }
    with open(dest_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def swap_in_outputs(stage: Path, processed_dir: Path, trash: Path) -> None:
    """Replace the previous dataset's preprocessing-owned entries with the staged ones.

    normalization.json is the launchers' "dataset complete" marker, so it is moved out first and
    moved in last (after the summary): an interruption inside this function leaves no marker and
    the next launcher run rebuilds. Entries preprocessing does not own are never touched.
    """
    owned = ("normalization.json", "preprocessing_summary.json", "train", "validation", "test")
    for name in owned:
        if (processed_dir / name).exists():
            os.replace(processed_dir / name, trash / name)
    for name in ("train", "validation", "test", "preprocessing_summary.json", "normalization.json"):
        os.replace(stage / name, processed_dir / name)
    stage.rmdir()
    shutil.rmtree(trash)


def _init_worker(level: str, parent_pid: int) -> None:
    """Pool-worker initializer: die with the parent (Linux prctl PR_SET_PDEATHSIG), then set up logging.

    A worker left behind by a killed parent would keep writing physical shards into the next
    run's _tmp_physical.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(err)}")
    # The parent may have died between fork and prctl, so no signal would ever come.
    if os.getppid() != parent_pid:
        raise SystemExit(f"parent {parent_pid} gone (ppid={os.getppid()})")
    _configure_logging(level)


def main() -> None:
    """Run the full preprocessing pipeline using the AUTOCHEM_CONFIG_PATH config.

    There is no implicit default config: the env var must point at an explicit
    config file (the same one used for training), keeping preprocessing and
    training perfectly in sync.
    """
    cfg_override = os.environ.get("AUTOCHEM_CONFIG_PATH", "").strip()
    if not cfg_override:
        raise RuntimeError(
            "AUTOCHEM_CONFIG_PATH is not set. Point it at an explicit config file, e.g.:\n"
            "  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u processing/preprocessing.py"
        )
    cfg_path = Path(cfg_override).expanduser().resolve()
    cfg_dict = load_json(cfg_path)
    validate_required_config_keys(cfg_dict)

    sys_cfg = _require_dict(cfg_dict, "system")
    log_level = _require_str(sys_cfg, "log_level")
    _configure_logging(log_level)

    cfg = parse_precfg(cfg_dict, cfg_path=cfg_path)

    raw_files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
    if not raw_files:
        raise FileNotFoundError(f"No raw HDF5 files found under {cfg.raw_dir} for patterns {cfg.raw_file_patterns}")

    out_final = cfg.processed_dir
    out_tmp = cfg.processed_dir / "_tmp_physical"
    out_stage = cfg.processed_dir / "_tmp_final"
    out_trash = cfg.processed_dir / "_tmp_old"

    # One run per processed_dir, taken before any filesystem change. The lock is a sibling of the
    # resolved processed_dir, so every spelling of the path (symlinks, relative components) shares
    # it. Closing lock_file releases the lock, so it is held in this frame for all of main();
    # forked workers inherit it.
    resolved = out_final.resolve()
    lock_path = resolved.parent / (resolved.name + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")  # held open for the whole run (the lock)
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            f"{lock_path} is held by another preprocessing run (or leftover workers of a killed run), or by a "
            "training or evaluation process reading this processed_dir (training holds it from its manifest "
            "read until both splits are preloaded); not touching it"
        ) from None

    # Refuse before creating or deleting anything: overwrite=false never deletes. The previous
    # dataset is not touched until the new one is complete (swap_in_outputs).
    if not cfg.overwrite:
        refuse_existing_outputs(out_final)
    for scratch in (out_tmp, out_stage, out_trash):
        clean_dir(scratch, overwrite=cfg.overwrite)

    try:
        _build_dataset(
            cfg,
            cfg_dict=cfg_dict,
            raw_files=raw_files,
            log_level=log_level,
            out_final=out_final,
            out_tmp=out_tmp,
            out_stage=out_stage,
            out_trash=out_trash,
        )
    except BaseException:
        log.error(
            "preprocessing failed; the previous dataset in %s is unchanged unless the swap had started "
            "(then normalization.json is absent); scratch dirs %s, %s, %s are left for inspection",
            out_final, out_tmp.name, out_stage.name, out_trash.name,
        )
        raise


def _check_unique_group_names(raw_files: List[Path], digests: Dict[int, np.ndarray]) -> None:
    """Raise if a group name occurs in two raw files (each copy would be split independently).

    Compares the workers' 64-bit name digests; a repeated digest is re-checked on the real names,
    so only a true duplicate raises (naming both files).
    """
    idx = np.concatenate([np.full(d.shape[0], i, dtype=np.int64) for i, d in sorted(digests.items())])
    allv = np.concatenate([d for _, d in sorted(digests.items())])
    order = np.argsort(allv, kind="stable")
    sv, si = allv[order], idx[order]
    rep = np.flatnonzero(sv[1:] == sv[:-1])
    pairs = sorted({(int(si[k]), int(si[k + 1])) for k in rep if si[k] != si[k + 1]})
    for a, b in pairs:
        with h5py.File(raw_files[a], "r") as fa, h5py.File(raw_files[b], "r") as fb:
            common = sorted(set(fa.keys()) & set(fb.keys()))
        if common:
            raise RuntimeError(
                f"group name(s) {common[:5]} (and {max(0, len(common) - 5)} more) occur in both "
                f"{raw_files[a].name} and {raw_files[b].name}: each copy would be assigned a split "
                "independently; remove the duplicate"
            )


def _merge_totals(rows: List[Dict[str, Any]], aggs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum the per-file counters (rows and aggregates, in file_idx order) into build totals."""
    tot: Dict[str, Any] = {
        "windows_written_by_split": {sp: sum(r["windows_written_by_split"][sp] for r in rows) for sp in _SPLITS},
        "groups_seen": sum(r["groups_seen"] for r in rows),
        "groups_kept": sum(r["groups_kept"] for r in rows),
        "group_rejects": {},
        "chunk_rejects": {},
        "frozen_tail": {
            k: {sp: sum(r["frozen_tail"][k][sp] for r in rows) for sp in _SPLITS}
            for k in ("truncated_groups", "removed_windows", "lost_groups")
        },
        "coverage": {},
        "drop_triggers": {},
        "drop_exact_zero": sum(a["drop_exact_zero"] for a in aggs),
        "t_start_hist": {sp: {} for sp in _SPLITS},
        "dt_hist": {sp: {} for sp in _SPLITS},
        "anchored": {sp: sum(a["anchored"][sp] for a in aggs) for sp in _SPLITS},
        "t_start_range": {},
    }
    for r in rows:
        for key in ("group_rejects", "chunk_rejects"):
            for k, v in r[key].items():
                tot[key][k] = tot[key].get(k, 0) + int(v)
    for a in aggs:
        for cell, c in a["coverage"].items():
            t = tot["coverage"].setdefault(cell, _new_cell())
            for k in ("too_few_valid", "reached_drop_test", "drop_below", "frozen_tail_no_window"):
                t[k] += c[k]
            for sp in _SPLITS:
                t["windows"][sp] += c["windows"][sp]
        for k, v in a["drop_triggers"].items():
            tot["drop_triggers"][k] = tot["drop_triggers"].get(k, 0) + v
        for key in ("t_start_hist", "dt_hist"):
            for sp in _SPLITS:
                for b, n in a[key][sp].items():
                    tot[key][sp][b] = tot[key][sp].get(b, 0) + n
        for sp, (lo, hi) in a["t_start_range"].items():
            cur = tot["t_start_range"].setdefault(sp, [lo, hi])
            cur[0], cur[1] = min(cur[0], lo), max(cur[1], hi)
    return tot


def _manifest_coverage(totals: Mapping[str, Any]) -> Dict[str, Any]:
    """normalization.json 'coverage': train windows per (T, log10 P) cell and the lowest-retention cell."""
    cells = sorted(totals["coverage"].items())
    worst = None
    for (ti, pi), c in cells:
        if c["reached_drop_test"] == 0:
            continue
        kept = c["reached_drop_test"] - c["drop_below"] - c["frozen_tail_no_window"]
        ret = kept / c["reached_drop_test"]
        if worst is None or ret < worst["retention"]:
            worst = {"T_lo_K": ti * _COVERAGE_T_BIN_K, "log10P_lo": pi * _COVERAGE_LOG10P_BIN_DEX,
                     "reached_drop_test": c["reached_drop_test"], "kept": kept, "retention": ret}
    return {
        "definition": (
            "train-split windows per floor-based (T, log10 P) cell (all splits' group counts are in "
            "preprocessing_summary.json); retention = groups kept / groups that reached the drop_below test "
            "(kept: not rejected as drop_below or frozen_tail_no_window)"
        ),
        "units": {"T": UNITS["T"], "P": UNITS["P"]},
        "T_bin_K": _COVERAGE_T_BIN_K,
        "log10P_bin_dex": _COVERAGE_LOG10P_BIN_DEX,
        "train_windows": [
            {"T_lo_K": ti * _COVERAGE_T_BIN_K, "log10P_lo": pi * _COVERAGE_LOG10P_BIN_DEX, "windows": c["windows"]["train"]}
            for (ti, pi), c in cells if c["windows"]["train"]
        ],
        "lowest_retention_cell": worst,
    }


def _build_dataset(
    cfg: PreCfg,
    *,
    cfg_dict: Mapping[str, Any],
    raw_files: List[Path],
    log_level: str,
    out_final: Path,
    out_tmp: Path,
    out_stage: Path,
    out_trash: Path,
) -> None:
    """Sample every raw file, normalize into out_stage, then swap the new dataset into out_final."""
    # Never more workers than files or than the cores this process may run on.
    n_workers = min(int(cfg.num_workers), len(raw_files), len(os.sched_getaffinity(0)))

    log.info(
        "Starting preprocessing raw_dir=%s processed_dir=%s files=%d num_workers=%d",
        str(cfg.raw_dir),
        str(cfg.processed_dir),
        len(raw_files),
        n_workers,
    )

    t0 = time.time()

    rows_by_idx: Dict[int, Dict[str, Any]] = {}
    aggs_by_idx: Dict[int, Dict[str, Any]] = {}

    # Fan out one process per raw file. Files are independent (per-file and per-trajectory RNGs,
    # per-file shard-name prefix), so there is no shared mutable state to guard.
    # fork is pinned: the initializer's parent-pid check assumes the pool's parent is this
    # process (under forkserver every worker's parent is the server). The initializer also
    # configures logging in each worker.
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_worker,
        initargs=(log_level, os.getpid()),
    ) as executor:
        futures = {
            executor.submit(sample_file, fp, idx, out_tmp=out_tmp, cfg=cfg): fp
            for idx, fp in enumerate(raw_files)
        }
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="raw files"):
                row, agg = fut.result()
                rows_by_idx[row["file_idx"]] = row
                aggs_by_idx[row["file_idx"]] = agg
                if row["n_pooled"] < row["n_groups"]:
                    log.warning(
                        "File %s: pool_size=%d drew %d of its %d source groups (deliberate subsample)",
                        futures[fut].name,
                        cfg.pool_size,
                        row["n_pooled"],
                        row["n_groups"],
                    )
                if row["truncated"]:
                    log.warning(
                        "File %s hit output_trajectories_per_file=%d (a cap on written windows, "
                        "%d per kept group); its remaining pooled source groups were not sampled",
                        futures[fut].name,
                        cfg.output_trajectories_per_file,
                        cfg.samples_per_source_trajectory,
                    )
        except BaseException:
            # Cancel queued files so a worker failure (or Ctrl+C) fails fast instead of
            # draining the whole queue. Pending futures must be cancelled directly: the
            # context manager's exit calls shutdown(wait=True, cancel_futures=False), which
            # resets the executor's cancel flag before the manager thread can act on it, so
            # shutdown(cancel_futures=True) alone is racy. Exit then only joins the in-flight
            # workers. SIGTERM is not handled: it kills this process outright, and the workers
            # then exit through PR_SET_PDEATHSIG.
            for pending_fut in futures:
                pending_fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    _check_unique_group_names(raw_files, {i: a.pop("name_digests") for i, a in aggs_by_idx.items()})
    rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
    totals = _merge_totals(rows, [aggs_by_idx[i] for i in sorted(aggs_by_idx)])
    counts_total = totals["windows_written_by_split"]

    total_written = counts_total["train"] + counts_total["validation"] + counts_total["test"]
    if total_written == 0:
        raise RuntimeError("No windows were written. Check group_rejects / chunk_rejects for the failure mode.")
    if totals["group_rejects"].get("non_positive_global", 0):
        log.warning(
            "%d trajectory groups rejected: P or T not > 0 (as float32); see group_rejects",
            totals["group_rejects"]["non_positive_global"],
        )

    log.info(
        "Totals: groups seen=%d kept=%d group_rejects=%s chunk_rejects=%s frozen_tail=%s windows=%d "
        "(train=%d val=%d test=%d)",
        totals["groups_seen"],
        totals["groups_kept"],
        totals["group_rejects"],
        totals["chunk_rejects"],
        totals["frozen_tail"],
        total_written,
        counts_total["train"],
        counts_total["validation"],
        counts_total["test"],
    )

    per_key_stats, methods, std_floored, species_sum = compute_train_stats(out_tmp, cfg)
    lo_hi = totals["t_start_range"].get("train")
    descriptive = {
        "window_start_time_s": {
            "definition": (
                "absolute time (s since the raw t=0) of the first state of the earliest and latest "
                "train-split window; earlier states never entered training (descriptive, not checkable "
                "by a host at call time)"
            ),
            "min": lo_hi[0] if lo_hi else None,
            "max": lo_hi[1] if lo_hi else None,
        },
        "coverage": _manifest_coverage(totals),
        "species_sum": species_sum,
        "std_floored": std_floored,
    }
    norm_sha256 = normalize_and_write(
        out_tmp=out_tmp,
        out_final=out_stage,
        cfg=cfg,
        per_key_stats=per_key_stats,
        methods=methods,
        counts_total=counts_total,
        descriptive=descriptive,
    )
    write_summary(
        out_stage, processed_dir=out_final, cfg=cfg, cfg_dict=cfg_dict, rows=rows, totals=totals, norm_sha256=norm_sha256
    )
    swap_in_outputs(out_stage, out_final, out_trash)
    remove_tmp_dir(out_tmp)
    log.info("Removed temporary physical shard directory: %s", out_tmp)

    log.info("Done in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
