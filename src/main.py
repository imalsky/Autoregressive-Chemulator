#!/usr/bin/env python3
"""
main.py - Strict entrypoint for training.

Principles (by design):
- No alias keys. Config must use the canonical schema.
- No silent fallbacks (no "search for a checkpoint", no "use CPU if CUDA missing", etc.).
- A failure prints its traceback and exits 1; a SIGTERM exits 143.

This script:
1) Loads config JSON and validates it (keys, types, cross-key rules, runtime constraints).
2) Resolves relative paths relative to the config file directory.
3) Loads the normalization manifest (processed_dir/normalization.json), builds the model
   and loads weights_only weights, all before the run directory is created.
4) Creates and flocks work_dir (one training process per run directory), then checks its
   contents (fresh: empty; resume: the run it continues) and opens work_dir/train.log.
5) Builds datasets/dataloaders sized for the rollouts actually used (memory budget first).
6) Runs Lightning training using explicit checkpoint behavior.

Notes:
- system.device ('cpu' or 'cuda') selects ONLY the dataset preload device (every split
  is preloaded). Training runs on runtime.accelerator ('cuda' or 'cpu') with exactly one
  device; on CUDA the launcher picks the GPU with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import fcntl
import logging
import logging.handlers
import math
import os
import signal
import traceback
import warnings
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.exceptions import SIGTERMException

from dataset import FlowMapRolloutDataset, create_dataloader, preload_nbytes
from model import _REQUIRED_GLOBALS, create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import (
    as_bool,
    as_int,
    as_str,
    atomic_write_json,
    dotted_differences,
    ensure_dir,
    is_comment_key,
    load_json_config,
    non_finite_tensor_keys,
    parse_precision_config,
    require,
    require_dict,
    require_dotted,
    require_same_model_block,
)

log = logging.getLogger(__name__)

# Shell convention for a process stopped by SIGTERM (143). Lightning turns SIGTERM into
# SIGTERMException, a SystemExit(None), which would otherwise exit 0.
_EXIT_STATUS_SIGTERM = 128 + int(signal.SIGTERM)

# Host-RAM budget for the dataset preload: daw is shared by ~22 users, so one training job may
# take at most half of the memory the kernel reports available at startup (value pending the
# author; MemAvailable is a snapshot on a shared host, not a guarantee).
_PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE = 0.5
_MEMINFO_PATH = Path("/proc/meminfo")
_BYTES_PER_KIB = 1024
_BYTES_PER_GB = 1e9

# Keys a resume may change: it continues the same run, so everything else must match the
# run's config.resolved.json (use checkpoint_mode='weights_only' in a new work_dir instead).
_RESUME_MUTABLE_KEYS = frozenset({"training.checkpoint_mode", "runtime.checkpoint", "training.max_epochs"})

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
# Run log in work_dir (appended on resume). A work_dir holding only this file counts as empty for
# a fresh start, so a run that failed after creating work_dir can simply be re-run.
_RUN_LOG_NAME = "train.log"
# Log records emitted before work_dir is locked are buffered (and also printed to stderr), then
# replayed into the run log once it is opened. Only the startup lines land here.
_STARTUP_LOG_BUFFER_RECORDS = 10_000
_startup_log_buffer: Optional[logging.handlers.MemoryHandler] = None
_run_log_handler: Optional[logging.FileHandler] = None
# fd of the exclusively flocked work_dir: held (never closed) for the process lifetime; the kernel
# releases the lock on any exit, including kill -9.
_work_dir_lock_fd: Optional[int] = None

# Rough checkpoint size for the save_top_k=-1 disk estimate: float32 live and EMA weights plus
# AdamW's two moment buffers.
_CKPT_BYTES_PER_PARAM = 16
_BYTES_PER_MB = 1e6

def _resolve_config_path() -> Path:
    """Resolve the training config path from the AUTOCHEM_CONFIG_PATH env var.

    There is no implicit default config. The env var must point at an explicit
    config file (e.g. ``configs/stage1.json``); this keeps every run reproducible
    and prevents silently training against a stale checked-in default.
    """
    cfg_override = os.environ.get("AUTOCHEM_CONFIG_PATH", "").strip()
    if not cfg_override:
        raise RuntimeError(
            "AUTOCHEM_CONFIG_PATH is not set. Point it at an explicit config file, e.g.:\n"
            "  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u src/main.py"
        )
    return Path(cfg_override).expanduser().resolve()

# Strict required config keys so saved config.resolved.json is always complete for debugging.
_REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
    "precision.compute_dtype",
    "precision.amp_mode",
    "precision.model_dtype",
    "precision.input_dtype",
    "precision.loss_dtype",
    "paths.raw_dir",
    "paths.processed_dir",
    "paths.work_dir",
    "normalization.epsilon",
    "normalization.min_std",
    "normalization.methods",
    "preprocessing.raw_file_patterns",
    "preprocessing.dt_min",
    "preprocessing.dt_max",
    "preprocessing.dt_sampling",
    "preprocessing.n_steps",
    "preprocessing.t_min",
    "preprocessing.output_trajectories_per_file",
    "preprocessing.shard_size",
    "preprocessing.overwrite",
    "preprocessing.time_key",
    "preprocessing.val_fraction",
    "preprocessing.test_fraction",
    "preprocessing.seed",
    "preprocessing.pool_size",
    "preprocessing.samples_per_source_trajectory",
    "preprocessing.max_chunk_attempts_per_source",
    "preprocessing.drop_below",
    "preprocessing.num_workers",
    "system.device",
    "system.log_level",
    "system.seed",
    "runtime.checkpoint",
    "runtime.accelerator",
    "runtime.devices",
    "runtime.strategy",
    "runtime.accumulate_grad_batches",
    "runtime.deterministic",
    "runtime.enable_progress_bar",
    "runtime.gradient_clip_val",
    "runtime.log_grad_norm",
    "runtime.log_every_n_steps",
    "runtime.checkpointing.enabled",
    "runtime.checkpointing.every_n_epochs",
    "runtime.checkpointing.monitor",
    "runtime.checkpointing.save_top_k",
    "runtime.checkpointing.save_last",
    "runtime.torch_compile.enabled",
    "runtime.torch_compile.backend",
    "runtime.torch_compile.mode",
    "runtime.torch_compile.dynamic",
    "runtime.torch_compile.fullgraph",
    "runtime.torch_compile.compile_forward_step",
    "runtime.torch_compile.compile_open_loop_unroll",
    "data.global_variables",
    "data.species_variables",
    "dataset.windows_per_trajectory",
    "model.type",
    "model.activation",
    "model.dropout",
    "model.layer_norm",
    "model.layer_norm_eps",
    "model.predict_delta",
    "model.mlp.hidden_dims",
    "model.mlp.residual",
    "training.batch_size",
    "training.max_epochs",
    "training.checkpoint_mode",
    "training.pin_memory",
    "training.rollout_steps",
    "training.loss.lambda_log10_mae",
    "training.loss.lambda_z_mse",
    "training.optimizer.name",
    "training.optimizer.lr",
    "training.optimizer.weight_decay",
    "training.optimizer.exclude_norm_and_bias_from_weight_decay",
    "training.optimizer.betas",
    "training.optimizer.eps",
    "training.optimizer.fused",
    "training.optimizer.foreach",
    "training.scheduler.enabled",
    "training.scheduler.warmup_epochs",
    # ReduceLROnPlateau (the only scheduler; linear warmup via warmup_epochs).
    "training.scheduler.factor",
    "training.scheduler.patience",
    "training.scheduler.threshold",
    "training.scheduler.min_lr",
    "training.scheduler.mode",
    "training.scheduler.monitor",
    "training.autoregressive_training.enabled",
    "training.autoregressive_training.skip_steps",
    "training.autoregressive_training.detach_between_steps",
    "training.autoregressive_training.backward_per_step",
    "training.curriculum.enabled",
    "training.curriculum.start_steps",
    "training.curriculum.end_steps",
    "training.curriculum.ramp_epochs",
    # EMA of weights (Lightning callback wired in trainer.py). Required so every
    # config states the policy explicitly (no silent default decay).
    "training.ema.enabled",
    "training.ema.decay",
    # Early stopping (Lightning callback wired in trainer.py). Required so every
    # config states the policy explicitly (no silent default patience/min_delta).
    "training.early_stopping.enabled",
    "training.early_stopping.patience",
    "training.early_stopping.min_delta",
)

# Optional keys accepted by schema validation.
_OPTIONAL_CONFIG_KEYS: Tuple[str, ...] = (
    # Optional fixed random-Fourier embedding of the normalized dt scalar
    # (Tancik et al. 2020). Absent or enabled=false -> plain scalar dt (legacy).
    "model.fourier_dt.enabled",
    "model.fourier_dt.num_freqs",
    "model.fourier_dt.sigma",
    "model.fourier_dt.seed",
    # Optional per-step discount gamma^(k-skip) on the autoregressive rollout loss.
    # Default 1.0 (uniform) preserves current behavior; only relevant when
    # autoregressive_training.enabled=true.
    "training.autoregressive_training.loss_discount_gamma",
)

# Mapping keys under these dotted paths are dynamic (validated elsewhere).
_OPEN_MAP_CONFIG_KEYS: Tuple[str, ...] = (
    "normalization.methods",
)

# Value kinds of every non-model key trainer.py reads, checked before any file is written or data
# is loaded (trainer.py's bool()/int()/float() casts would otherwise turn "false" into True and
# 3.7 epochs into 3). bool/int are exact types; float accepts int or float, finite, never bool;
# opt_float may be null; float_pair is a 2-list of floats. Optional keys are checked when present.
# (runtime.devices / strategy / accumulate_grad_batches / accelerator: _validate_runtime_constraints.)
_CONFIG_KEY_KINDS: Dict[str, str] = {
    "training.max_epochs": "int",
    "training.rollout_steps": "int",
    "training.autoregressive_training.enabled": "bool",
    "training.autoregressive_training.skip_steps": "int",
    "training.autoregressive_training.detach_between_steps": "bool",
    "training.autoregressive_training.backward_per_step": "bool",
    "training.autoregressive_training.loss_discount_gamma": "float",
    "training.curriculum.enabled": "bool",
    "training.curriculum.start_steps": "int",
    "training.curriculum.end_steps": "int",
    "training.curriculum.ramp_epochs": "int",
    "training.loss.lambda_log10_mae": "float",
    "training.loss.lambda_z_mse": "float",
    "training.optimizer.name": "str",
    "training.optimizer.lr": "float",
    "training.optimizer.weight_decay": "float",
    "training.optimizer.exclude_norm_and_bias_from_weight_decay": "bool",
    "training.optimizer.betas": "float_pair",
    "training.optimizer.eps": "float",
    "training.optimizer.fused": "bool",
    "training.optimizer.foreach": "bool",
    "training.scheduler.enabled": "bool",
    "training.scheduler.warmup_epochs": "int",
    "training.scheduler.factor": "float",
    "training.scheduler.patience": "int",
    "training.scheduler.threshold": "float",
    "training.scheduler.min_lr": "float",
    "training.scheduler.mode": "str",
    "training.scheduler.monitor": "str",
    "training.ema.enabled": "bool",
    "training.ema.decay": "float",
    "training.early_stopping.enabled": "bool",
    "training.early_stopping.patience": "int",
    "training.early_stopping.min_delta": "float",
    "runtime.gradient_clip_val": "opt_float",
    "runtime.log_grad_norm": "bool",
    "runtime.log_every_n_steps": "int",
    "runtime.enable_progress_bar": "bool",
    "runtime.deterministic": "bool",
    "runtime.checkpointing.enabled": "bool",
    "runtime.checkpointing.every_n_epochs": "int",
    "runtime.checkpointing.monitor": "str",
    "runtime.checkpointing.save_top_k": "int",
    "runtime.checkpointing.save_last": "bool",
    "runtime.torch_compile.enabled": "bool",
    "runtime.torch_compile.backend": "str",
    "runtime.torch_compile.mode": "str",
    "runtime.torch_compile.dynamic": "bool",
    "runtime.torch_compile.fullgraph": "bool",
    "runtime.torch_compile.compile_forward_step": "bool",
    "runtime.torch_compile.compile_open_loop_unroll": "bool",
}

# Explicit devices only: Lightning's "gpu" alias tries MPS first and "auto" falls back to CPU
# silently; a missing GPU must be an error.
_ACCELERATORS: Tuple[str, ...] = ("cuda", "cpu")
_PRELOAD_DEVICES: Tuple[str, ...] = ("cpu", "cuda")


def _build_allowed_config_prefixes() -> set[str]:
    """Expand every known dotted key into the set of allowed key prefixes (for unknown-key checks)."""
    out: set[str] = set()
    keys = list(_REQUIRED_CONFIG_KEYS) + list(_OPTIONAL_CONFIG_KEYS) + list(_OPEN_MAP_CONFIG_KEYS)
    for dotted in keys:
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return out


_ALLOWED_CONFIG_PREFIXES = _build_allowed_config_prefixes()
_OPEN_MAP_KEY_SET = set(_OPEN_MAP_CONFIG_KEYS)


def _validate_no_unknown_config_keys(mapping: Mapping[str, Any], *, prefix: str = "") -> None:
    """Recursively reject any key not present in the allowed-prefix set ('_comment*' keys excepted).

    Dotted raw keys are rejected too: nested blocks are the only spelling of a nested key.
    """
    for raw_key, val in mapping.items():
        if not isinstance(raw_key, str):
            raise TypeError("bad config key type")
        if raw_key.strip() != raw_key:
            raise ValueError(f"config key has leading/trailing whitespace: {raw_key!r} under {prefix or '<root>'}")

        # Comments are allowed anywhere in the config tree; any other '_' key is unknown.
        if is_comment_key(raw_key):
            continue
        # A dotted raw key ("a.b": v) would read as the nested key a.b but is never looked up.
        if "." in raw_key:
            raise KeyError(f"ambiguous config key (contains '.'): {prefix + '.' if prefix else ''}{raw_key}")

        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if path not in _ALLOWED_CONFIG_PREFIXES:
            raise KeyError(f"unknown config key: {path}")

        if path in _OPEN_MAP_KEY_SET:
            if not isinstance(val, Mapping):
                raise TypeError(f"bad type: {path}")
            continue

        if isinstance(val, Mapping):
            _validate_no_unknown_config_keys(val, prefix=path)


def _is_finite_number(v: Any) -> bool:
    """int or float (never bool) and finite."""
    return type(v) in (int, float) and math.isfinite(v)


def _validate_config_key_kinds(cfg: Mapping[str, Any]) -> None:
    """Check the value kind of every key in _CONFIG_KEY_KINDS (optional keys only when present)."""
    for key, kind in _CONFIG_KEY_KINDS.items():
        try:
            v = require_dotted(cfg, key)
        except KeyError:
            if key in _OPTIONAL_CONFIG_KEYS:
                continue
            raise
        if kind == "bool":
            ok = type(v) is bool
        elif kind == "int":
            ok = type(v) is int
        elif kind == "float":
            ok = _is_finite_number(v)
        elif kind == "opt_float":
            ok = v is None or _is_finite_number(v)
        elif kind == "str":
            ok = isinstance(v, str) and bool(v.strip())
        elif kind == "float_pair":
            ok = isinstance(v, list) and len(v) == 2 and all(_is_finite_number(x) for x in v)
        else:
            raise RuntimeError(f"unknown kind {kind!r} for {key} (bug)")
        if not ok:
            raise TypeError(f"bad type: {key} = {v!r} (expected {kind})")


def validate_required_config_keys(cfg: Mapping[str, Any]) -> None:
    """Strictly validate the full training config before anything is written or loaded.

    Required keys, no unknown keys, value kinds, runtime constraints and cross-key rules.
    """
    for key in _REQUIRED_CONFIG_KEYS:
        require_dotted(cfg, key)
    _validate_no_unknown_config_keys(cfg)
    _validate_config_key_kinds(cfg)
    _validate_runtime_constraints(cfg)
    _validate_value_ranges(cfg)
    _validate_cross_key_constraints(cfg)


def _validate_runtime_constraints(cfg: Mapping[str, Any]) -> None:
    """Enforce single-device, no-accumulation invariants and explicit devices.

    The autoregressive manual-optimization path in trainer.py was never validated against
    multi-process strategies or gradient accumulation, so those configs hard-fail here, as do
    'auto' device choices and a CUDA request with no visible GPU.
    """
    runtime = require_dict(cfg, "runtime")

    accum = runtime.get("accumulate_grad_batches")
    if type(accum) is not int or accum != 1:
        raise ValueError(
            "runtime.accumulate_grad_batches must be 1 (gradient accumulation is unsupported)"
        )

    devices = runtime.get("devices")
    if type(devices) is not int or devices != 1:
        raise ValueError(
            f"runtime.devices must be the integer 1, got {devices!r} (one device; multi-GPU/DDP is "
            "unsupported; the launcher picks the GPU with CUDA_VISIBLE_DEVICES)"
        )

    strategy = runtime.get("strategy")
    if strategy != "auto":
        raise ValueError(
            'runtime.strategy must be "auto" (DDP / multi-process strategies are unsupported)'
        )

    accelerator = runtime.get("accelerator")
    if accelerator not in _ACCELERATORS:
        raise ValueError(f"runtime.accelerator must be one of {list(_ACCELERATORS)}, got {accelerator!r}")

    preload_device = require_dict(cfg, "system").get("device")
    if preload_device not in _PRELOAD_DEVICES:
        raise ValueError(f"system.device must be one of {list(_PRELOAD_DEVICES)}, got {preload_device!r}")

    if "cuda" in (accelerator, preload_device) and not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device visible; check CUDA_VISIBLE_DEVICES "
            f"(runtime.accelerator={accelerator!r}, system.device={preload_device!r})"
        )

    # Pinning applies to host tensors: the preload must live in host RAM for pin_memory=true.
    if preload_device == "cuda" and require_dotted(cfg, "training.pin_memory") is True:
        raise ValueError("training.pin_memory=true requires system.device='cpu' (the preload is pinned host memory)")


def _validate_value_ranges(cfg: Mapping[str, Any]) -> None:
    """Single-key value ranges (kinds already checked), so a bad value fails before any write or preload.

    Each Lightning sentinel has one meaning here: gradient_clip_val null is the only "no clipping";
    max_epochs -1 ("no limit") is unsupported.
    """
    tcfg = require_dict(cfg, "training")
    max_epochs = tcfg["max_epochs"]
    if max_epochs < 1:
        raise ValueError(
            f"training.max_epochs must be >= 1, got {max_epochs} (Lightning's -1 'no epoch limit' is "
            "unsupported; early stopping with a large max_epochs gives that behaviour)"
        )

    clip = require_dict(cfg, "runtime")["gradient_clip_val"]
    if clip is not None and not clip > 0:
        raise ValueError(f"runtime.gradient_clip_val must be null (no clipping) or > 0, got {clip}")

    opt = tcfg["optimizer"]
    if opt["name"] != "adamw":
        raise ValueError(f"training.optimizer.name must be 'adamw' (the only optimizer), got {opt['name']!r}")
    if not all(0.0 <= b < 1.0 for b in opt["betas"]):
        raise ValueError(f"training.optimizer.betas must both be in [0, 1), got {opt['betas']}")
    for key in ("lr", "eps"):
        if not opt[key] > 0:
            raise ValueError(f"training.optimizer.{key} must be > 0, got {opt[key]}")
    if not opt["weight_decay"] >= 0:
        raise ValueError(f"training.optimizer.weight_decay must be >= 0, got {opt['weight_decay']}")
    if opt["fused"] and opt["foreach"]:
        raise ValueError("training.optimizer.fused and training.optimizer.foreach cannot both be true")

    sched = tcfg["scheduler"]
    if sched["warmup_epochs"] < 0:
        raise ValueError(f"training.scheduler.warmup_epochs must be >= 0, got {sched['warmup_epochs']}")
    if sched["enabled"]:
        if not 0.0 < sched["factor"] < 1.0:
            raise ValueError(f"training.scheduler.factor must be in (0, 1), got {sched['factor']}")
        for key in ("patience", "threshold", "min_lr"):
            if not sched[key] >= 0:
                raise ValueError(f"training.scheduler.{key} must be >= 0, got {sched[key]}")


def _validate_cross_key_constraints(cfg: Mapping[str, Any]) -> None:
    """Curriculum, rollout and checkpointing rules that span several keys (kinds already checked)."""
    tcfg = require_dict(cfg, "training")
    rollout = tcfg["rollout_steps"]
    max_epochs = tcfg["max_epochs"]
    cur = tcfg["curriculum"]
    ar = tcfg["autoregressive_training"]

    if cur["enabled"]:
        start, end, ramp = cur["start_steps"], cur["end_steps"], cur["ramp_epochs"]
        if not 1 <= start <= end == rollout:
            raise ValueError(
                "training.curriculum requires 1 <= start_steps <= end_steps == training.rollout_steps, "
                f"got start_steps={start}, end_steps={end}, rollout_steps={rollout}"
            )
        if not 1 <= ramp < max_epochs:
            raise ValueError(
                f"training.curriculum.ramp_epochs must satisfy 1 <= ramp_epochs < training.max_epochs, "
                f"got ramp_epochs={ramp}, max_epochs={max_epochs}"
            )

    if ar["enabled"]:
        # The epoch-0 training horizon: the curriculum starts below rollout_steps.
        k_min = min(rollout, cur["start_steps"]) if cur["enabled"] else rollout
        skip = ar["skip_steps"]
        if not 0 <= skip < k_min:
            horizon = f"curriculum.start_steps={cur['start_steps']}; " if cur["enabled"] else ""
            raise ValueError(
                f"training.autoregressive_training.skip_steps ({skip}) must be >= 0 and < the epoch-0 "
                f"training horizon K={k_min} ({horizon}rollout_steps={rollout})"
            )

    ckpt = require_dict(require_dict(cfg, "runtime"), "checkpointing")
    if ckpt["enabled"]:
        top_k, every = ckpt["save_top_k"], ckpt["every_n_epochs"]
        if not (top_k == -1 or top_k >= 1):
            raise ValueError(f"runtime.checkpointing.save_top_k must be -1 or >= 1, got {top_k}")
        if not 1 <= every <= max_epochs:
            raise ValueError(
                f"runtime.checkpointing.every_n_epochs must satisfy 1 <= every_n_epochs <= training.max_epochs, "
                f"got {every} (max_epochs={max_epochs})"
            )


def _apply_compute_globals(cfg: Mapping[str, Any]) -> None:
    """Set process-wide matmul/TF32 precision flags consistent with runtime.deterministic.

    TF32 is forced off under deterministic=True; otherwise the reproducibility contract would
    be silently broken regardless of Lightning's own deterministic flag.
    """
    runtime = require_dict(cfg, "runtime")
    deterministic = as_bool(require(runtime, "deterministic"), "runtime.deterministic")
    if deterministic:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _config_dir(cfg_path: Path) -> Path:
    """Return the directory that relative config paths are resolved against (the config's parent)."""
    return cfg_path.parent


def _resolve_path(root: Path, p: str) -> str:
    """Resolve a possibly-relative path string against `root`; absolute paths pass through."""
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((root / pth).resolve())


def _to_relative_path_str(p: str, *, start: Path) -> str:
    """
    Persist path-like config fields as relative to `start` for portability.
    """
    pth = Path(str(p).strip()).expanduser()
    if not pth.is_absolute():
        return str(pth)
    try:
        return os.path.relpath(str(pth.resolve()), str(start.resolve()))
    except ValueError:
        # os.path.relpath raises ValueError only for cross-drive paths on Windows.
        return str(pth)


def _portable_config_snapshot(cfg: Mapping[str, Any], *, save_dir: Path) -> Dict[str, Any]:
    """
    Build a config snapshot suitable for disk persistence.

    Runtime uses absolute resolved paths internally; for portability we save
    path-like fields relative to the run directory.
    """
    out = dict(cfg)

    paths_raw = out.get("paths")
    if isinstance(paths_raw, Mapping):
        paths_out: Dict[str, Any] = dict(paths_raw)
        for k, v in paths_out.items():
            if isinstance(v, str) and v.strip():
                paths_out[k] = _to_relative_path_str(v, start=save_dir)
        out["paths"] = paths_out

    runtime_raw = out.get("runtime")
    if isinstance(runtime_raw, Mapping):
        runtime_out: Dict[str, Any] = dict(runtime_raw)
        ckpt = runtime_out.get("checkpoint")
        if isinstance(ckpt, str) and ckpt.strip():
            runtime_out["checkpoint"] = _to_relative_path_str(ckpt, start=save_dir)
        out["runtime"] = runtime_out

    return out


def resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Resolve cfg.paths[*] and runtime.checkpoint relative to the config file directory.

    runtime.checkpoint may be null (no checkpoint); null is preserved as-is.
    Absolutizing it here keeps the persisted config snapshot on a single base:
    _portable_config_snapshot re-bases all absolute path fields to the run directory.
    """
    root = _config_dir(cfg_path)

    out = dict(cfg)
    paths = require_dict(out, "paths")
    resolved: Dict[str, Any] = dict(paths)

    for k, v in paths.items():
        if isinstance(v, str) and v.strip():
            resolved[k] = _resolve_path(root, v)

    out["paths"] = resolved

    runtime = require_dict(out, "runtime")
    runtime_out: Dict[str, Any] = dict(runtime)
    ckpt = runtime_out.get("checkpoint")
    if ckpt is not None:
        runtime_out["checkpoint"] = _resolve_path(root, as_str(ckpt, "runtime.checkpoint"))
    out["runtime"] = runtime_out
    return out


def configure_logging(cfg: Mapping[str, Any]) -> None:
    """Initialize root logging from system.log_level and route warnings through logging.

    Lightning's loggers are routed through the root handler too: importing lightning (before
    this runs) gives the 'lightning' logger its own handler with propagate=False, so its lines
    would ignore system.log_level and the timestamp format. The parent 'lightning' logger must be
    included, or every Lightning line prints twice.

    Records are also buffered until _attach_run_log replays them into work_dir/train.log.
    """
    global _startup_log_buffer
    sys_cfg = require_dict(cfg, "system")
    level_str = str(require(sys_cfg, "log_level")).upper().strip()
    levels = logging.getLevelNamesMapping()
    if level_str not in levels:
        raise ValueError(f"bad system.log_level: {level_str!r} (expected one of {sorted(levels)})")
    level = levels[level_str]
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    logging.captureWarnings(True)
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(level)
    # No target yet and a flush level above CRITICAL: the buffer only collects until the run log opens.
    _startup_log_buffer = logging.handlers.MemoryHandler(
        _STARTUP_LOG_BUFFER_RECORDS, flushLevel=logging.CRITICAL + 1
    )
    logging.getLogger().addHandler(_startup_log_buffer)


def _attach_run_log(work_dir: Path) -> None:
    """Append every later log record to work_dir/train.log and replay the buffered startup records."""
    global _run_log_handler, _startup_log_buffer
    handler = logging.FileHandler(work_dir / _RUN_LOG_NAME, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if _startup_log_buffer is not None:
        root.removeHandler(_startup_log_buffer)
        _startup_log_buffer.setTarget(handler)
        _startup_log_buffer.flush()
        _startup_log_buffer.close()
        _startup_log_buffer = None
    _run_log_handler = handler


def _write_run_log_only(level: int, msg: str, *, exc_info: Any = None) -> None:
    """Write one record to train.log alone (stderr already shows it), if the run log is open."""
    if _run_log_handler is None:
        return
    _run_log_handler.handle(log.makeRecord(log.name, level, __file__, 0, msg, (), exc_info))


def _lock_work_dir(work_dir: Path) -> None:
    """Take an exclusive, non-blocking flock on the work_dir directory itself, held until exit.

    A second training process on the same work_dir fails here, before any state check or write.
    flock (not lockf, which gives EBADF on a directory fd); the fd is never closed.
    """
    global _work_dir_lock_fd
    fd = os.open(work_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"work_dir is in use by another training process: {work_dir}") from None
    _work_dir_lock_fd = fd


def configure_runtime_warning_filters() -> None:
    """Suppress known false-positive Lightning warnings (preloaded batch stream, CSV log directory)."""
    # CSVLogger is rooted at work_dir, which is never empty when it starts; the only file Lightning
    # deletes there is metrics.csv, which cannot exist then (fresh: empty work_dir; resume: rotated to
    # metrics.pre_restart*.csv first; a concurrent launch is excluded by the work_dir lock).
    warnings.filterwarnings(
        "ignore",
        message=r".*Experiment logs directory .* exists and is not empty.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'train_dataloader' does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'val_dataloader' does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Your `IterableDataset` has `__len__` defined\..*",
    )


def select_preload_device(cfg: Mapping[str, Any]) -> torch.device:
    """Dataset preload device from system.device ('cpu' | 'cuda'; checked in _validate_runtime_constraints)."""
    return torch.device(as_str(require(require_dict(cfg, "system"), "device"), "system.device"))


# ==============================================================================
# Manifest consistency
# ==============================================================================


def _require_unique_species(species: List[str], source: str) -> None:
    """Species names index the per-key statistics and the output columns: each must appear once."""
    dups = sorted({s for s in species if species.count(s) > 1})
    if dups:
        raise ValueError(f"{source}: duplicate species names {dups}")


def load_manifest_and_validate_config(
    cfg: Mapping[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load normalization.json and require that it declares species_variables and
    global_variables exactly matching config.data (missing keys or any mismatch is a hard error)."""
    mpath = processed_dir / "normalization.json"
    if not mpath.exists():
        raise FileNotFoundError(
            f"missing normalization.json in {processed_dir} (paths.processed_dir; relative paths "
            "resolve against the config file directory)"
        )

    manifest = load_json_config(mpath)

    data_cfg = require_dict(cfg, "data")
    cfg_species = require(data_cfg, "species_variables")
    cfg_globals = require(data_cfg, "global_variables")

    if not isinstance(cfg_species, list) or not all(isinstance(x, str) for x in cfg_species) or not cfg_species:
        raise TypeError("bad data.species_variables")
    if not isinstance(cfg_globals, list) or not all(isinstance(x, str) for x in cfg_globals):
        raise TypeError("bad data.global_variables")
    if list(cfg_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"data.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")
    _require_unique_species(cfg_species, "data.species_variables")

    if "species_variables" not in manifest:
        raise KeyError("normalization.json missing species_variables")
    if "global_variables" not in manifest:
        raise KeyError("normalization.json missing global_variables")

    man_species = manifest["species_variables"]
    man_globals = manifest["global_variables"]

    if not isinstance(man_species, list) or not all(isinstance(x, str) for x in man_species) or not man_species:
        raise TypeError("bad manifest.species_variables")
    _require_unique_species(man_species, f"{mpath} species_variables")
    if list(cfg_species) != list(man_species):
        raise ValueError(f"species_variables mismatch: config={cfg_species} manifest={man_species}")

    if not isinstance(man_globals, list) or not all(isinstance(x, str) for x in man_globals):
        raise TypeError("bad manifest.global_variables")
    if list(cfg_globals) != list(man_globals):
        raise ValueError(f"global_variables mismatch: config={cfg_globals} manifest={man_globals}")

    return manifest, list(cfg_species), list(cfg_globals)


# ==============================================================================
# Rollout sizing
# ==============================================================================


def max_rollout_steps_for_training(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for the training split (accounts for the curriculum end horizon)."""
    base_k = as_int(require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError(f"training.rollout_steps must be >= 1, got {base_k}")

    cur = require_dict(tcfg, "curriculum")
    enabled = as_bool(require(cur, "enabled"), "training.curriculum.enabled")
    if not enabled:
        return base_k

    start_steps = as_int(require(cur, "start_steps"), "training.curriculum.start_steps")
    end_steps = as_int(require(cur, "end_steps"), "training.curriculum.end_steps")
    if start_steps < 1 or end_steps < 1:
        raise ValueError(
            f"training.curriculum.start_steps and end_steps must be >= 1, got start_steps={start_steps}, "
            f"end_steps={end_steps}"
        )

    return int(max(base_k, end_steps))


def max_rollout_steps_for_eval(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for validation split (fixed horizon)."""
    base_k = as_int(require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError(f"training.rollout_steps must be >= 1, got {base_k}")
    return base_k


# ==============================================================================
# Dataloaders
# ==============================================================================


def _mem_available_bytes() -> int:
    """MemAvailable from /proc/meminfo, in bytes (Linux; daw is the only host)."""
    for line in _MEMINFO_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * _BYTES_PER_KIB
    raise RuntimeError(f"MemAvailable not found in {_MEMINFO_PATH}")


def _check_preload_budget(processed_dir: Path, preload_device: torch.device) -> None:
    """Log the train+validation preload size (shard headers only) and refuse an oversize CPU preload.

    Runs before either split is loaded: host allocation is lazy, so an oversize preload would
    otherwise be OOM-killed midway through filling. A CUDA preload fails at allocation with the
    byte count, so only the host-RAM case is checked.
    """
    need = 0
    for split in ("train", "validation"):
        n_traj, nbytes = preload_nbytes(processed_dir, split)
        log.info("preload %s: %d trajectories, %.3f GB", split, n_traj, nbytes / _BYTES_PER_GB)
        need += nbytes
    if preload_device.type != "cpu":
        return
    avail = _mem_available_bytes()
    cap = _PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE * avail
    log.info(
        "preload train+validation: %.3f GB (cap %.3f GB = %.2f x MemAvailable %.3f GB)",
        need / _BYTES_PER_GB, cap / _BYTES_PER_GB, _PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE, avail / _BYTES_PER_GB,
    )
    if need > cap:
        raise RuntimeError(
            f"dataset preload needs {need / _BYTES_PER_GB:.1f} GB of host RAM for train+validation, above the "
            f"cap of {_PRELOAD_MAX_FRACTION_OF_MEM_AVAILABLE} x MemAvailable ({avail / _BYTES_PER_GB:.1f} GB) = "
            f"{cap / _BYTES_PER_GB:.1f} GB; use a smaller processed dataset (there is no loading path for data "
            "larger than the budget)"
        )


def create_dataloaders(
    cfg: Mapping[str, Any],
    *,
    tcfg: Mapping[str, Any],
    preload_device: torch.device,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val dataloaders (strict), after the preload memory budget check."""
    paths = require_dict(cfg, "paths")
    processed_dir = Path(as_str(require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    ds_cfg = require_dict(cfg, "dataset")
    windows_per_traj = as_int(require(ds_cfg, "windows_per_trajectory"), "dataset.windows_per_trajectory")

    batch_size = as_int(require(tcfg, "batch_size"), "training.batch_size")
    pin_memory = as_bool(require(tcfg, "pin_memory"), "training.pin_memory")

    if batch_size <= 0:
        raise ValueError("bad batch_size")

    k_train = max_rollout_steps_for_training(tcfg)
    k_eval = max_rollout_steps_for_eval(tcfg)

    _check_preload_budget(processed_dir, preload_device)

    # Every split is preloaded (single-process batch stream) as float32 tensors (the shard
    # dtype) on the preload device.
    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        device=preload_device,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_eval, seed=seed + 1, **common_ds_kwargs)

    # The training stream is reseeded per epoch (resume-safe); validation keeps one continuing stream.
    train_dl = create_dataloader(train_ds, batch_size=batch_size, pin_memory=pin_memory, drop_last=True, epoch_seeded=True)
    val_dl = create_dataloader(val_ds, batch_size=batch_size, pin_memory=pin_memory, drop_last=False, epoch_seeded=False)

    log.info("Dataloaders: train(K=%d) val(K=%d)", k_train, k_eval)
    return train_dl, val_dl


# ==============================================================================
# Checkpoints (strict)
# ==============================================================================


def _load_weights_only(module: FlowMapRolloutModule, ckpt_path: Path) -> None:
    """Load Lightning checkpoint weights only (no optimizer/scheduler state), strictly.

    Uses ``ema_state_dict`` (saved by EMACallback), which holds the smoothed weights that
    produced the checkpoint's val_loss; it is required when the checkpoint's own run had EMA
    enabled. The checkpoint's saved model block must equal this run's (comment keys ignored),
    every floating tensor must be finite, and any missing or unexpected key makes
    load_state_dict raise.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError("ckpt not found")

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise TypeError(f"not a Lightning checkpoint (no 'state_dict'): {ckpt_path}")
    hparams = obj.get("hyper_parameters")
    if not isinstance(hparams, Mapping) or "training" not in hparams:
        raise KeyError(f"checkpoint has no hyper_parameters['training'] block: {ckpt_path}")
    require_same_model_block(hparams.get("model"), module.hparams["model"], source=str(ckpt_path))

    if "ema_state_dict" in obj:
        source = "ema_state_dict"
    elif bool(hparams["training"]["ema"]["enabled"]):
        raise KeyError(f"checkpoint run had EMA enabled but holds no ema_state_dict: {ckpt_path}")
    else:
        source = "state_dict"
    state_dict = obj[source]
    if not isinstance(state_dict, dict):
        raise TypeError("bad state_dict")

    bad = non_finite_tensor_keys(state_dict)
    if bad:
        raise ValueError(f"checkpoint {ckpt_path} has non-finite weights: {bad[:3]}")

    module.load_state_dict(state_dict, strict=True)
    log.info("weights_only: loaded %s from %s", source, ckpt_path)


def _log_keep_all_checkpoints_estimate(cfg: Mapping[str, Any], total_params: int) -> None:
    """runtime.checkpointing.save_top_k=-1 keeps every checkpoint: log the disk it may take."""
    ckpt = cfg["runtime"]["checkpointing"]
    if not (ckpt["enabled"] and ckpt["save_top_k"] == -1):
        return
    n_ckpt = int(cfg["training"]["max_epochs"]) // int(ckpt["every_n_epochs"])
    per_ckpt = total_params * _CKPT_BYTES_PER_PARAM
    log.info(
        "runtime.checkpointing.save_top_k=-1 keeps every checkpoint: up to %d x ~%.0f MB = ~%.1f GB "
        "(estimate: float32 live and EMA weights plus two AdamW moments per parameter)",
        n_ckpt, per_ckpt / _BYTES_PER_MB, n_ckpt * per_ckpt / _BYTES_PER_GB,
    )


def _check_resume_checkpoint(ckpt_path: Path, work_dir: Path, *, checkpointing: bool) -> None:
    """A resume must continue this run: its checkpoint lives in, and was written to, work_dir/checkpoints.

    Lightning restores ModelCheckpoint state only when the recorded dirpath equals the current
    one; a moved or copied run dir would silently drop best/last tracking.
    """
    ckpt_dir = (work_dir / "checkpoints").resolve()
    if ckpt_path.resolve().parent != ckpt_dir:
        raise ValueError(
            f"resume continues the run in paths.work_dir; runtime.checkpoint must be in {ckpt_dir} "
            f"(got {ckpt_path}); to start a new run from another run's weights use "
            "training.checkpoint_mode='weights_only'"
        )
    if not checkpointing:
        return
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    callbacks = obj.get("callbacks") if isinstance(obj, dict) else None
    states = [v for k, v in (callbacks or {}).items() if str(k).startswith("ModelCheckpoint")]
    if not states:
        raise ValueError(f"resume checkpoint {ckpt_path} holds no ModelCheckpoint state; best/last tracking would reset")
    expected = os.path.realpath(ckpt_dir)
    for state in states:
        stored = state.get("dirpath") if isinstance(state, Mapping) else None
        if stored != expected:
            raise ValueError(
                f"resume checkpoint {ckpt_path} was written to {stored!r}, not {expected!r}; resuming a "
                "moved, copied or foreign run would drop best/last checkpoint tracking (restart it with "
                "checkpoint_mode='weights_only' in a new work_dir)"
            )


def _check_resume_config(cfg: Mapping[str, Any], work_dir: Path) -> None:
    """A resume must use the run's own config: compare with config.resolved.json (comment keys ignored)."""
    snapshot = work_dir / "config.resolved.json"
    if not snapshot.is_file():
        raise FileNotFoundError(f"resume requires the run's config snapshot: {snapshot}")
    diff = [
        k for k in dotted_differences(load_json_config(snapshot), _portable_config_snapshot(cfg, save_dir=work_dir))
        if k not in _RESUME_MUTABLE_KEYS
    ]
    if diff:
        raise ValueError(
            f"resume config differs from {snapshot} at: {', '.join(diff)}. A resume continues the same run "
            f"(only {', '.join(sorted(_RESUME_MUTABLE_KEYS))} may change); to change these, start a new run "
            "with training.checkpoint_mode='weights_only'"
        )


def _next_free_path(work_dir: Path, stem: str, suffix: str) -> Path:
    """work_dir/<stem><suffix>, or <stem>.<i><suffix> with the first free i >= 1."""
    dst = work_dir / f"{stem}{suffix}"
    i = 1
    while dst.exists():
        dst = work_dir / f"{stem}.{i}{suffix}"
        i += 1
    return dst


def main() -> None:
    """Entry point: load + validate the config, build datasets/model, and run Lightning training.

    Config path comes from AUTOCHEM_CONFIG_PATH (required). Checkpoint behavior follows
    training.checkpoint_mode (none / resume / weights_only) strictly, with no implicit search.
    The config, manifest, model and weights_only checkpoint are checked before work_dir is
    created; then work_dir is locked against a second training process and every check of its
    contents (fresh: empty or only train.log; resume: the run it continues) and every write
    happens under that lock.
    """
    cfg_path = _resolve_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    cfg = load_json_config(cfg_path)
    validate_required_config_keys(cfg)

    cfg = resolve_paths(cfg, cfg_path)
    configure_logging(cfg)
    _apply_compute_globals(cfg)

    # Required top-level sections (strict).
    tcfg = require_dict(cfg, "training")
    runtime = require_dict(cfg, "runtime")
    sys_cfg = require_dict(cfg, "system")
    paths = require_dict(cfg, "paths")

    seed = as_int(require(sys_cfg, "seed"), "system.seed")
    seed_everything(seed, workers=True)

    work_dir = Path(as_str(require(paths, "work_dir"), "paths.work_dir")).expanduser().resolve()
    ckpt_enabled = as_bool(
        require(require_dict(runtime, "checkpointing"), "enabled"),
        "runtime.checkpointing.enabled",
    )

    # Check checkpoint_mode early to decide if work_dir must be empty.
    ckpt_mode = as_str(require(tcfg, "checkpoint_mode"), "training.checkpoint_mode").lower()
    if ckpt_mode not in ("none", "resume", "weights_only"):
        raise ValueError(f"bad training.checkpoint_mode: {ckpt_mode!r} (expected none/resume/weights_only)")

    # Validate the checkpoint config before any expensive work (dataset preload,
    # model/trainer build) and before work_dir is created or written to, so a
    # bad config fails in seconds and leaves no partial run directory behind.
    ckpt_val = require(runtime, "checkpoint")  # required key; may be null; absolute via resolve_paths
    ckpt_path: Optional[Path] = None
    if ckpt_val is not None:
        ckpt_path = Path(as_str(ckpt_val, "runtime.checkpoint"))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        if not ckpt_path.is_file():
            raise ValueError(f"checkpoint must be a file: {ckpt_path}")

    if ckpt_mode == "none" and ckpt_path is not None:
        raise ValueError("checkpoint_mode=none with checkpoint set")
    if ckpt_mode == "resume" and ckpt_path is None:
        raise ValueError("resume requires checkpoint")
    if ckpt_mode == "weights_only" and ckpt_path is None:
        raise ValueError("weights_only requires checkpoint")

    processed_dir = Path(as_str(require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    # Manifest must exist and must declare variable lists that exactly match config.data.
    manifest, species_vars, global_vars = load_manifest_and_validate_config(cfg, processed_dir)

    prec = parse_precision_config(cfg)
    log.info(
        "precision: compute=%s amp=%s model=%s input=%s loss=%s lightning=%s (preload float32)",
        str(prec.compute_dtype),
        prec.amp_mode,
        str(prec.model_dtype),
        str(prec.input_dtype),
        str(prec.loss_dtype),
        prec.lightning_precision,
    )

    model = create_model(dict(cfg))
    model = model.to(dtype=prec.model_dtype)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: type=%s S=%d G=%d params=%d trainable=%d",
             str(require(require_dict(cfg, "model"), "type")),
             len(species_vars), len(global_vars), total_params, trainable_params)

    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
        precision=prec,
    )
    if ckpt_mode == "weights_only":
        _load_weights_only(lit_module, ckpt_path)
    _log_keep_all_checkpoints_estimate(cfg, total_params)

    ensure_dir(work_dir)
    _lock_work_dir(work_dir)

    # No automatic backups. Fresh training requires an empty work_dir (a lone train.log from a
    # run that failed after creating work_dir counts as empty and is appended to).
    # Resume mode allows non-empty work_dir (continuing in same directory).
    if ckpt_mode != "resume":
        if any(p.name != _RUN_LOG_NAME for p in work_dir.iterdir()):
            raise RuntimeError(
                f"work_dir not empty: {work_dir}. "
                "Fresh training (checkpoint_mode != 'resume') requires an empty directory. "
                "Either clear/move its contents, point paths.work_dir elsewhere, or set "
                "training.checkpoint_mode='resume' and runtime.checkpoint=<work_dir>/checkpoints/last.ckpt "
                "to continue in-place."
            )
    else:
        # A resume continues this run: its own checkpoint, its own ModelCheckpoint state, its own config.
        _check_resume_checkpoint(ckpt_path, work_dir, checkpointing=ckpt_enabled)
        _check_resume_config(cfg, work_dir)
    _attach_run_log(work_dir)

    preload_device = select_preload_device(cfg)
    log.info("preload device: %s", preload_device)

    configure_runtime_warning_filters()

    train_dl, val_dl = create_dataloaders(
        cfg,
        tcfg=tcfg,
        preload_device=preload_device,
        seed=seed,
    )
    try:
        train_batches_per_epoch: Optional[int] = len(train_dl)
    except TypeError:
        train_batches_per_epoch = None

    # On resume runs, preserve any existing metrics.csv before Lightning touches it.
    # (Non-resume modes require an empty work_dir, so nothing can exist yet.)
    metrics_part: Optional[Path] = None
    if ckpt_mode == "resume":
        m = work_dir / "metrics.csv"
        if m.exists():
            metrics_part = _next_free_path(work_dir, "metrics.pre_restart", ".csv")
            m.replace(metrics_part)

    trainer = build_lightning_trainer(
        cfg,
        work_dir=work_dir,
        precision_config=prec,
        train_batches_per_epoch=train_batches_per_epoch,
    )
    # config.resolved.json is write-once: it records the config the run was started with (for
    # stage 2, the handoff source). A resume records its own config next to it.
    if ckpt_mode == "resume":
        snapshot_path = _next_free_path(work_dir, "config.resume", ".json")
    else:
        snapshot_path = work_dir / "config.resolved.json"
    atomic_write_json(snapshot_path, _portable_config_snapshot(cfg, save_dir=work_dir))

    trainer.fit(
        lit_module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=str(ckpt_path) if ckpt_mode == "resume" else None,
    )

    # A resume that trained no further epoch writes no metrics.csv: move the part back.
    if metrics_part is not None and not (work_dir / "metrics.csv").exists():
        metrics_part.replace(work_dir / "metrics.csv")

    # Publish the best-val_loss checkpoint under a stable name for export tooling
    # (testing/export.py, testing/aoti_export.py consume checkpoints/best.ckpt).
    if ckpt_enabled and trainer.is_global_zero:
        top_k = [c for c in trainer.checkpoint_callbacks if getattr(c, "monitor", None) == "val_loss"]
        if len(top_k) != 1:
            raise RuntimeError(f"expected exactly one val_loss ModelCheckpoint callback, found {len(top_k)}")
        best_src = str(top_k[0].best_model_path or "").strip()
        if not best_src:
            raise RuntimeError(
                "checkpointing enabled but ModelCheckpoint.best_model_path is empty; "
                "no best checkpoint was saved during fit"
            )
        best_path = Path(best_src)
        if not best_path.is_file():
            raise FileNotFoundError(f"best checkpoint not found: {best_path}")
        ckpt_dir = (work_dir / "checkpoints").resolve()
        if best_path.resolve().parent != ckpt_dir:
            raise RuntimeError(
                f"best checkpoint {best_path} is outside this run's checkpoints dir {ckpt_dir}; "
                "refusing to publish it as best.ckpt"
            )
        dst = work_dir / "checkpoints" / "best.ckpt"
        tmp = dst.parent / (dst.name + ".tmp")
        shutil.copy2(best_path, tmp)
        tmp.replace(dst)
        log.info("best checkpoint (val_loss): %s -> %s", best_path, dst)


if __name__ == "__main__":
    try:
        main()
    except SIGTERMException:
        # Lightning raises this (a SystemExit) after the SIGTERM'd epoch step; best.ckpt is not published.
        msg = (
            "terminated by SIGTERM before training finished; checkpoints/best.ckpt was not published or "
            "updated by this run; resume from checkpoints/last.ckpt"
        )
        print(msg, file=sys.stderr)
        _write_run_log_only(logging.ERROR, msg)
        raise SystemExit(_EXIT_STATUS_SIGTERM)
    except Exception:
        traceback.print_exc()
        _write_run_log_only(logging.ERROR, "training failed", exc_info=sys.exc_info())
        raise SystemExit(1)
