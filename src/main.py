#!/usr/bin/env python3
"""
main.py - Strict entrypoint for training.

Principles (by design):
- No alias keys. Config must use the canonical schema.
- No silent fallbacks (no "search for a checkpoint", no "use CPU if CUDA missing", etc.).
- Errors are concise and deterministic.

This script:
1) Loads config JSON.
2) Resolves relative paths relative to the config file directory.
3) Loads normalization manifest (processed_dir/normalization.json) and checks config consistency.
4) Builds datasets/dataloaders sized for the rollouts actually used.
5) Builds model strictly from model.type.
6) Runs Lightning training using explicit checkpoint behavior.

Notes:
- system.device selects ONLY the dataset preload device (every split is preloaded).
  Lightning decides training device(s) based on cfg.runtime.
"""

from __future__ import annotations

import logging
import os
import warnings
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from lightning.pytorch import seed_everything

from dataset import FlowMapRolloutDataset, create_dataloader
from model import _REQUIRED_GLOBALS, create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import (
    PrecisionConfig,
    as_bool,
    as_int,
    as_str,
    atomic_write_json,
    ensure_dir,
    load_json_config,
    parse_precision_config,
    require,
    require_dict,
    require_dotted,
)

log = logging.getLogger(__name__)

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
    "precision.preload_dtype",
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
    # Optional perf knob for manual-optimization training path (default in code if absent).
    "runtime.log_grad_norm",
    # Number of worker processes for the preprocessing per-file sampling fan-out.
    # Consumed only by processing/preprocessing.py; harmless/optional to the trainer.
    "preprocessing.num_workers",
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
    """Recursively reject any non-comment key not present in the allowed-prefix set."""
    for raw_key, val in mapping.items():
        if not isinstance(raw_key, str):
            raise TypeError("bad config key type")
        if raw_key.strip() != raw_key:
            raise KeyError("ambiguous config key whitespace")

        # Comments are allowed anywhere in the config tree.
        if raw_key.startswith("_"):
            continue

        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if path not in _ALLOWED_CONFIG_PREFIXES:
            raise KeyError(f"unknown config key: {path}")

        if path in _OPEN_MAP_KEY_SET:
            if not isinstance(val, Mapping):
                raise TypeError(f"bad type: {path}")
            continue

        if isinstance(val, Mapping):
            _validate_no_unknown_config_keys(val, prefix=path)


def validate_required_config_keys(cfg: Mapping[str, Any]) -> None:
    """Strictly validate the full training config: required keys, no unknown keys, runtime constraints."""
    for key in _REQUIRED_CONFIG_KEYS:
        require_dotted(cfg, key)
    _validate_no_unknown_config_keys(cfg)
    _validate_runtime_constraints(cfg)


def _validate_runtime_constraints(cfg: Mapping[str, Any]) -> None:
    """Enforce single-device, no-accumulation invariants (DDP / grad-accum are unsupported).

    The autoregressive manual-optimization path in trainer.py was never validated against
    multi-process strategies or gradient accumulation, so those configs hard-fail here.
    """
    runtime = require_dict(cfg, "runtime")

    accum = runtime.get("accumulate_grad_batches")
    if accum != 1:
        raise ValueError(
            "runtime.accumulate_grad_batches must be 1 (gradient accumulation is unsupported)"
        )

    devices = runtime.get("devices")
    if devices not in ("auto", 1):
        raise ValueError(
            'runtime.devices must be "auto" or 1 (multi-GPU is unsupported)'
        )

    strategy = runtime.get("strategy")
    if strategy != "auto":
        raise ValueError(
            'runtime.strategy must be "auto" (DDP / multi-process strategies are unsupported)'
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
    """Initialize root logging from system.log_level and route warnings through logging."""
    sys_cfg = require_dict(cfg, "system")
    level_str = str(require(sys_cfg, "log_level")).upper().strip()
    level = getattr(logging, level_str, None)
    if level is None:
        raise ValueError("bad log_level")
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.captureWarnings(True)


def configure_runtime_warning_filters() -> None:
    """Suppress known false-positive Lightning warnings for the single-process preloaded batch stream."""
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
    """Select the dataset preload device from system.device (cpu | auto | cuda[:N])."""
    sys_cfg = require_dict(cfg, "system")
    pref = str(require(sys_cfg, "device")).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        return torch.device(pref)

    raise ValueError("bad system.device")


# ==============================================================================
# Manifest consistency
# ==============================================================================


def load_manifest_and_validate_config(
    cfg: Mapping[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load normalization.json and require that it declares species_variables and
    global_variables exactly matching config.data (missing keys or any mismatch is a hard error)."""
    mpath = processed_dir / "normalization.json"
    if not mpath.exists():
        raise FileNotFoundError("missing normalization.json")

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

    if "species_variables" not in manifest:
        raise KeyError("normalization.json missing species_variables")
    if "global_variables" not in manifest:
        raise KeyError("normalization.json missing global_variables")

    man_species = manifest["species_variables"]
    man_globals = manifest["global_variables"]

    if not isinstance(man_species, list) or not all(isinstance(x, str) for x in man_species) or not man_species:
        raise TypeError("bad manifest.species_variables")
    if list(cfg_species) != list(man_species):
        raise ValueError("species_variables mismatch")

    if not isinstance(man_globals, list) or not all(isinstance(x, str) for x in man_globals):
        raise TypeError("bad manifest.global_variables")
    if list(cfg_globals) != list(man_globals):
        raise ValueError("global_variables mismatch")
    if list(man_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"manifest.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")

    return manifest, list(cfg_species), list(cfg_globals)


# ==============================================================================
# Rollout sizing
# ==============================================================================


def max_rollout_steps_for_training(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for the training split (accounts for the curriculum end horizon)."""
    base_k = as_int(require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")

    cur = require_dict(tcfg, "curriculum")
    enabled = as_bool(require(cur, "enabled"), "training.curriculum.enabled")
    if not enabled:
        return base_k

    start_steps = as_int(require(cur, "start_steps"), "training.curriculum.start_steps")
    end_steps = as_int(require(cur, "end_steps"), "training.curriculum.end_steps")
    if start_steps < 1 or end_steps < 1:
        raise ValueError("bad curriculum steps")

    return int(max(base_k, end_steps))


def max_rollout_steps_for_eval(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for validation split (fixed horizon)."""
    base_k = as_int(require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")
    return base_k


# ==============================================================================
# Dataloaders
# ==============================================================================


def create_dataloaders(
    cfg: Mapping[str, Any],
    *,
    tcfg: Mapping[str, Any],
    precision: PrecisionConfig,
    preload_device: torch.device,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val dataloaders (strict)."""
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

    # Every split is preloaded (single-process batch stream); tensors are stored as
    # precision.preload_dtype on the preload device.
    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        device=preload_device,
        storage_dtype=precision.preload_dtype,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_eval, seed=seed + 1, **common_ds_kwargs)

    train_dl = create_dataloader(train_ds, batch_size=batch_size, pin_memory=pin_memory, drop_last=True)
    val_dl = create_dataloader(val_ds, batch_size=batch_size, pin_memory=pin_memory, drop_last=False)

    log.info("Dataloaders: train(K=%d) val(K=%d)", k_train, k_eval)
    return train_dl, val_dl


# ==============================================================================
# Checkpoints (strict)
# ==============================================================================


def _load_weights_only(module: torch.nn.Module, ckpt_path: Path) -> None:
    """Load Lightning checkpoint weights only (no optimizer/scheduler state), strictly.

    Prefers ``ema_state_dict`` when present (saved by EMACallback) since it
    holds the smoothed weights that produced the checkpoint's val_loss. Any
    missing or unexpected key makes torch's load_state_dict raise.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError("ckpt not found")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("bad ckpt")

    state_dict = obj.get("ema_state_dict")
    if not (isinstance(state_dict, dict) and state_dict):
        if "state_dict" not in obj:
            raise KeyError("missing state_dict")
        state_dict = obj["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("bad state_dict")

    module.load_state_dict(state_dict, strict=True)


def main() -> None:
    """Entry point: load + validate the config, build datasets/model, and run Lightning training.

    Config path comes from AUTOCHEM_CONFIG_PATH (required). Checkpoint behavior follows
    training.checkpoint_mode (none / resume / weights_only) strictly, with no implicit search.
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

    # Check checkpoint_mode early to decide if work_dir must be empty.
    ckpt_mode = as_str(require(tcfg, "checkpoint_mode"), "training.checkpoint_mode").lower()
    if ckpt_mode not in ("none", "resume", "weights_only"):
        raise ValueError("bad checkpoint_mode")

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

    # No automatic backups. Fresh training requires an empty work_dir.
    # Resume mode allows non-empty work_dir (continuing in same directory).
    if ckpt_mode != "resume":
        if work_dir.exists() and any(work_dir.iterdir()):
            raise RuntimeError(
                f"work_dir not empty: {work_dir}. "
                "Fresh training (checkpoint_mode != 'resume') requires an empty directory. "
                "Either clear/move its contents, point paths.work_dir elsewhere, or set "
                "training.checkpoint_mode='resume' to continue in-place."
            )
    ensure_dir(work_dir)

    processed_dir = Path(as_str(require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    # Manifest must exist and must declare variable lists that exactly match config.data.
    manifest, species_vars, global_vars = load_manifest_and_validate_config(cfg, processed_dir)

    preload_device = select_preload_device(cfg)
    log.info("preload device: %s", preload_device)

    configure_runtime_warning_filters()

    prec = parse_precision_config(cfg)
    log.info(
        "precision: compute=%s amp=%s model=%s input=%s preload=%s loss=%s lightning=%s",
        str(prec.compute_dtype),
        prec.amp_mode,
        str(prec.model_dtype),
        str(prec.input_dtype),
        str(prec.preload_dtype),
        str(prec.loss_dtype),
        prec.lightning_precision,
    )

    train_dl, val_dl = create_dataloaders(
        cfg,
        tcfg=tcfg,
        precision=prec,
        preload_device=preload_device,
        seed=seed,
    )
    try:
        train_batches_per_epoch: Optional[int] = len(train_dl)
    except TypeError:
        train_batches_per_epoch = None

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

    # On resume runs, preserve any existing metrics.csv before Lightning touches it.
    # (Non-resume modes require an empty work_dir, so nothing can exist yet.)
    if ckpt_mode == "resume" and os.environ.get("LOCAL_RANK", "0") == "0":
        m = work_dir / "metrics.csv"
        if m.exists():
            dst = work_dir / "metrics.pre_restart.csv"
            i = 1
            while dst.exists():
                dst = work_dir / f"metrics.pre_restart.{i}.csv"
                i += 1
            m.replace(dst)

    trainer = build_lightning_trainer(
        cfg,
        work_dir=work_dir,
        precision_config=prec,
        train_batches_per_epoch=train_batches_per_epoch,
    )
    atomic_write_json(work_dir / "config.resolved.json", _portable_config_snapshot(cfg, save_dir=work_dir))

    if ckpt_mode == "none":
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

    elif ckpt_mode == "resume":
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=str(ckpt_path))

    elif ckpt_mode == "weights_only":
        _load_weights_only(lit_module, ckpt_path)
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

    # Publish the best-val_loss checkpoint under a stable name for export tooling
    # (testing/export.py, testing/aoti_export.py consume checkpoints/best.ckpt).
    ckpt_enabled = as_bool(
        require(require_dict(runtime, "checkpointing"), "enabled"),
        "runtime.checkpointing.enabled",
    )
    if ckpt_enabled and trainer.is_global_zero:
        mc = trainer.checkpoint_callback
        if mc is None:
            raise RuntimeError("checkpointing enabled but no ModelCheckpoint callback found")
        best_src = str(mc.best_model_path or "").strip()
        if not best_src:
            raise RuntimeError(
                "checkpointing enabled but ModelCheckpoint.best_model_path is empty; "
                "no best checkpoint was saved during fit"
            )
        best_path = Path(best_src)
        if not best_path.is_file():
            raise FileNotFoundError(f"best checkpoint not found: {best_path}")
        dst = work_dir / "checkpoints" / "best.ckpt"
        tmp = dst.parent / (dst.name + ".tmp")
        shutil.copy2(best_path, tmp)
        tmp.replace(dst)
        log.info("best checkpoint (val_loss): %s -> %s", best_path, dst)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Concise top-level error. No traceback by default.
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
