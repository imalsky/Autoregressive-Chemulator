#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export.py - Export a self-contained 1-step PHYSICAL-space model (dynamic batch).

Goal:
- Produce portable artifact(s) that can be loaded and run without carrying the processed dataset folder around.
- Bake normalization (species, globals, dt) into the exported module.
- Support dynamic batch size B (shape-polymorphic) for all inputs.
- Intended for inference: model is exported in eval() mode with gradients disabled.

Exported module signature (PHYSICAL units, the training data's units; metadata 'units'):
  y_next = model(y_phys, dt_seconds, g_phys)

Where:
  y_phys     : [B, S]  species mixing ratios relative to the whole gas (n_i / n_total over all gas of
                       the generator run, untracked species included, not renormalized over the tracked
                       species); training compositions contain no He, so He-bearing compositions are
                       outside the validated domain; the training initial gas held only the tracked
                       species, so they sum to 1 at t=0 and at most 1 later (metadata 'species_sum');
                       values <= epsilon are floored at epsilon before log10; species are not range-checked
  dt_seconds : [B]     time step in s; any in-range dt per call and per row (metadata 'dt_validity':
                       multi-step rollouts are validated only at constant dt within a rollout)
  g_phys     : [B, G]  G = 2 mandatory: P in dyn/cm^2 (barye), T in K

Output:
  y_next_phys: [B, S]  same denominator as the input; neither clipped nor renormalized

Out-of-range rows (metadata oor_policy 'nan_row'): a row whose dt, P or T lies outside the train-split
range recorded in the metadata ('oor_box', in the normalizer's coordinate: log10 for dt and log-*
methods) by more than OOR_TOL of the range, or whose inputs are not finite, comes back as NaN in
every species. A NaN row means only that; in-range rows are computed exactly as without the guard.

Saved .pt2 includes an embedded JSON metadata blob via torch.export.save(extra_files=...).
Read it back with:
  extra = {"metadata.json": ""}
  ep = torch.export.load("...pt2", extra_files=extra)
  meta = json.loads(extra["metadata.json"])

Run (the run directory is paths.work_dir of the config; it must hold COMPLETED, i.e. a finished run):
  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/export.py

Behavior:
- Exports every target in EXPORT_DEVICES ("cpu,cuda"); a requested target that is not available is
  an error (set EXPORT_DEVICES = "cpu" to export CPU only), so a stale artifact is never left
  silently next to a fresh one.
- Each artifact is gated before it reaches its final name: written under .partial/, reloaded, and
  checked on its export device against the eager wrapper (in-range rows finite and equal within
  EXPORT_PARITY_RTOL, out-of-range rows NaN), then renamed into place.
- The checkpoint's normalization fingerprint (norm_sha256) must match the processed data's.
- Re-exporting the same checkpoint with the same code and torch build reproduces the file's sha256:
  the example inputs are seeded and node stack traces (absolute checkout paths) are stripped.

Notes:
- For correctness, export uses strict=True by default.
- Exports are float32 only (a bf16 export was never numerically validated).
- Each .pt2 is pinned to its export device: torch.export bakes device=<export device> into
  the input asserts, so .to(device) on ep.module() is not enough. On GPU, load
  export_cuda_dynB_1step_phys.pt2.
- build_export() is shared with testing/aoti_export.py, so both artifacts carry the same
  graph, weights and metadata.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.export import Dim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import _REQUIRED_GLOBALS, create_model  # noqa: E402
from run_dir import resolve_run_dir  # noqa: E402
from utils import (  # noqa: E402
    file_sha256,
    git_state,
    load_json_config,
    non_finite_tensor_keys,
    normalization_sha256,
    require_same_model_block,
)

# Runtime settings (edit here; no argparse). The run directory comes from AUTOCHEM_CONFIG_PATH
# (testing/run_dir.py).
# Best-val_loss checkpoint (written by src/main.py after training); relative paths are
# resolved against the run config directory and the file must exist.
CHECKPOINT = "checkpoints/best.ckpt"
# Optional processed-dir override (absolute or repo-relative) for run dirs whose configured
# processed_dir has been relocated. When set, it is the sole candidate and must contain
# normalization.json. None => use cfg.paths.processed_dir strictly (no fallbacks).
PROCESSED_DIR_OVERRIDE: Optional[str] = None
EXPORT_DEVICES = "cpu,cuda"  # requested targets; every one must be available (no fallback)
EXPORT_STRICT = True
EXAMPLE_BATCH = 4
# torch.export specializes a dynamic dim whose example size is 1 (torch 2.8).
MIN_EXAMPLE_BATCH = 2
B_MIN = 1
# Largest exported batch. Inherited with no recorded rationale and kept pending the host envelope
# (Canoe's per-rank cell count is unknown). The .pt2 raises above it (torch.export
# input constraints, checked by the export gate); the AOTI package does not check B (its host guard
# does; testing/aoti_export.py). Dim("B", min=1) without a max was verified bit-identical to eager.
B_MAX = 16384
# Example inputs only fix the traced shapes; seeding them makes the .pt2 byte-reproducible.
EXAMPLE_INPUT_SEED = 0
# The export gate's own in-range draws (independent of the example inputs).
EXPORT_VERIFY_SEED = 1
# Export-vs-eager tolerance of the gate (relative, atol 0): CPU is bitwise today; the headroom is for
# CUDA kernels, to be re-checked on the first trained CUDA export.
EXPORT_PARITY_RTOL = 1e-6
# Out-of-range rule (see the module docstring): allowed excursion outside the train-split range, as
# a fraction of each input's normalized range; absorbs the float32 round trip at the exact edges.
OOR_TOL = 1e-6
# metadata_version 2 added units, the trained box and NaN-row policy, fingerprints and provenance,
# and removed cuda_visible_devices; 'format' is unchanged. Bump it only when a key is removed or
# changes meaning: an added key (dt_validity) keeps the version. The units 'species' string is copied
# from the build's normalization.json (processing/preprocessing.py UNITS), so an older build carries
# the wording it was built with.
METADATA_VERSION = 2
# What the one-step export's dt input is validated for (docs/DECISIONS.md open item 21, decision (b)).
DT_VALIDITY = (
    "the one-step export accepts any in-range dt per call and per row; multi-step rollouts are "
    "trained and validated only at constant dt within a rollout; rollouts whose dt changes between "
    "steps are unvalidated"
)

# Checkpoint state-dict key prefixes (FlowMapRolloutModule): the network under "model.", the
# training-only loss buffers under "criterion." (dropped).
_MODEL_KEY_PREFIX = "model."
_LOSS_KEY_PREFIX = "criterion."
# The metric best.ckpt was selected by; must equal src/trainer.py _MONITOR_METRIC (not imported:
# trainer pulls in Lightning).
_MONITOR_METRIC = "val_loss"
# A finished artifact is written under <dir>/.partial/<same name> and os.replace'd onto its final
# name, so a failed or killed export never leaves a truncated file under the final name.
_PARTIAL_DIR_NAME = ".partial"
# Written by src/main.py only when a run finished; export refuses a run without it.
_COMPLETED_NAME = "COMPLETED"
# Manifest keys the metadata copies; a processed dir without them must be rebuilt.
_REQUIRED_MANIFEST_KEYS = ("units", "window_start_time_s", "coverage")


# =============================================================================
# Small device diagnostics
# =============================================================================


def _cuda_available() -> bool:
    """True if a usable CUDA device is present."""
    return torch.cuda.is_available()


def _print_device_diag() -> None:
    """Print a one-line summary of torch version and CUDA availability."""
    cuda_ok = _cuda_available()
    print(
        "Device diag:"
        f" torch={torch.__version__}"
        f" torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        f" cuda.is_available={cuda_ok}"
        f" cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    if cuda_ok:
        try:
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass


# =============================================================================
# IO helpers
# =============================================================================


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object strictly (no duplicate keys, no NaN/Infinity; src/utils.load_json_config)."""
    return load_json_config(path)


def save_artifact(final_path: Path, write: Any, *, verify: Any) -> None:
    """Run write(staging_path), then verify(staging_path), then move it onto final_path atomically.

    The staging path is final_path's name under <dir>/.partial/ (same filesystem; torch requires the
    .pt2 suffix and names the package's internal root after the file's basename, so the artifact
    is identical to one written in place). The staging file is removed if write or verify fails,
    so a failed artifact never replaces the previous one.
    """
    staging = final_path.parent / _PARTIAL_DIR_NAME / final_path.name
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(staging)
        verify(staging)
        os.replace(staging, final_path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    if not any(staging.parent.iterdir()):
        staging.parent.rmdir()


def _resolve_path_like(raw: str, *, base: Path) -> Path:
    """Resolve a possibly-relative path string against `base`; absolute paths are resolved too (symlinks followed)."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        return (base / p).resolve()
    return p.resolve()


def _to_repo_relative_str(path: Path) -> str:
    """Return `path` relative to the repo root for portable metadata."""
    return os.path.relpath(str(path.resolve()), str(REPO_ROOT.resolve()))


def _load_resolved_config(run_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """
    Require the resolved training config from the run directory.
    """
    cfg_path = run_dir / "config.resolved.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {cfg_path}")
    return _load_json(cfg_path), cfg_path


def _resolve_checkpoint_path(raw: str, *, cfg_path: Path) -> Path:
    """
    Resolve a checkpoint path strictly:
      - relative paths are resolved against config directory
      - checkpoint must exist and be a file
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (cfg_path.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    if not p.is_file():
        raise ValueError(f"checkpoint must be a file: {p}")
    return p


# =============================================================================
# Checkpoint loading (strict)
# =============================================================================


def _load_weights_strict(
    model: nn.Module, ckpt_path: Path, *, expected_model: Mapping[str, Any], norm_sha256: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a Lightning checkpoint's network weights into `model` strictly.

    - Only a Lightning checkpoint (a mapping with 'state_dict' and hyper_parameters) is accepted.
    - Its normalization fingerprint (hyper_parameters['norm_sha256']) must equal `norm_sha256`, the
      fingerprint of the normalization.json being baked (no fingerprint = refused).
    - Its saved model block must equal `expected_model` (comment keys ignored): shape mismatches
      fail in load_state_dict anyway, this also catches activation, layer_norm_eps, dropout.
    - 'ema_state_dict' (saved by EMACallback; the weights that produced the checkpoint's
      val_loss) is used when present and required when the checkpoint's run had EMA enabled
      (mirrors src/main.py:_load_weights_only).
    - Every key must start with 'model.' (one prefix stripped) or 'criterion.' (training-only
      loss buffers, dropped); every floating tensor must be finite.

    Returns (checkpoint record for the metadata, the checkpoint's model block without comment keys).
    The record holds weights_source, epoch, global_step, monitor, checkpoint_score (the val_loss
    ModelCheckpoint's current_score: this checkpoint's own score) and training_lightning_version.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, Mapping) or "state_dict" not in ckpt:
        raise TypeError(f"not a Lightning checkpoint (no 'state_dict'): {ckpt_path}")
    hparams = ckpt.get("hyper_parameters")
    if not isinstance(hparams, Mapping) or "training" not in hparams:
        raise KeyError(f"checkpoint has no hyper_parameters['training'] block: {ckpt_path}")
    if hparams.get("norm_sha256") != norm_sha256:
        raise ValueError(
            f"checkpoint {ckpt_path} norm_sha256={hparams.get('norm_sha256')!r} does not match the "
            f"normalization.json being exported ({norm_sha256}): it was trained on other normalization "
            "statistics (or before the fingerprint existed)"
        )
    model_block = require_same_model_block(hparams.get("model"), expected_model, source=str(ckpt_path))

    if "ema_state_dict" in ckpt:
        source = "ema_state_dict"
    elif bool(hparams["training"]["ema"]["enabled"]):
        raise KeyError(f"checkpoint run had EMA enabled but holds no ema_state_dict: {ckpt_path}")
    else:
        source = "state_dict"
    state = ckpt[source]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {source} must be a mapping, got {type(state).__name__}")
    print(f"Using {source} from checkpoint.")

    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        if k.startswith(_LOSS_KEY_PREFIX):
            continue
        if not k.startswith(_MODEL_KEY_PREFIX):
            raise KeyError(f"unexpected checkpoint key {k!r} (expected 'model.*' or 'criterion.*'): {ckpt_path}")
        cleaned[k[len(_MODEL_KEY_PREFIX):]] = v

    bad = non_finite_tensor_keys(cleaned)
    if bad:
        raise ValueError(f"checkpoint {ckpt_path} has non-finite weights: {bad[:3]}")

    model.load_state_dict(cleaned, strict=True)

    score = None
    for key, state in (ckpt.get("callbacks") or {}).items():
        if str(key).startswith("ModelCheckpoint") and f"'monitor': {_MONITOR_METRIC!r}" in str(key):
            v = state.get("current_score")
            score = float(v) if v is not None else None
    record = {
        "weights_source": source,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "monitor": _MONITOR_METRIC,
        "checkpoint_score": score,
        "training_lightning_version": ckpt.get("pytorch-lightning_version"),
    }
    return record, model_block


def _freeze_for_inference(model: nn.Module) -> nn.Module:
    """Put the model in eval mode and disable grads on all parameters (in place)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# Normalization bake-in (must match preprocessing.py exactly)
# =============================================================================


def _canonical_method(method: str) -> str:
    """Normalize a normalization-method name to lowercase/stripped form."""
    m = str(method).lower().strip()
    return m


_SUPPORTED_SPECIES_METHODS = {"log-standard"}
_SUPPORTED_GLOBAL_METHODS = {"standard", "min-max", "log-min-max", "log-standard"}


class BakedNormalizer(nn.Module):
    """
    Torch module containing all normalization constants as buffers.

    This is intentionally simple and explicit (no dependencies on a separate normalizer.py),
    and should match processing/preprocessing.py behavior.
    """

    def __init__(
        self,
        *,
        species_vars: Sequence[str],
        global_vars: Sequence[str],
        species_methods: Sequence[str],
        global_methods: Sequence[str],
        s_log_mean: torch.Tensor,  # [S]
        s_log_std: torch.Tensor,   # [S]
        s_log_min: torch.Tensor,   # [S]
        s_log_max: torch.Tensor,   # [S]
        s_eps: torch.Tensor,       # [S]
        g_mean: torch.Tensor,      # [G]
        g_std: torch.Tensor,       # [G]
        g_min: torch.Tensor,       # [G]
        g_max: torch.Tensor,       # [G]
        g_log_mean: torch.Tensor,  # [G]
        g_log_std: torch.Tensor,   # [G]
        g_log_min: torch.Tensor,   # [G]
        g_log_max: torch.Tensor,   # [G]
        g_eps: torch.Tensor,       # [G]
        dt_log_min: float,
        dt_log_max: float,
        dt_eps: float,
    ) -> None:
        super().__init__()

        self.species_vars = tuple(species_vars)
        self.global_vars = tuple(global_vars)
        self.species_methods = tuple(species_methods)
        self.global_methods = tuple(global_methods)

        # Species (log-space stats)
        self.register_buffer("s_log_mean", s_log_mean.to(torch.float32))
        self.register_buffer("s_log_std", s_log_std.to(torch.float32))
        self.register_buffer("s_log_min", s_log_min.to(torch.float32))
        self.register_buffer("s_log_max", s_log_max.to(torch.float32))
        self.register_buffer("s_eps", s_eps.to(torch.float32))

        # Globals (raw + log-space min/max)
        self.register_buffer("g_mean", g_mean.to(torch.float32))
        self.register_buffer("g_std", g_std.to(torch.float32))
        self.register_buffer("g_min", g_min.to(torch.float32))
        self.register_buffer("g_max", g_max.to(torch.float32))
        self.register_buffer("g_log_mean", g_log_mean.to(torch.float32))
        self.register_buffer("g_log_std", g_log_std.to(torch.float32))
        self.register_buffer("g_log_min", g_log_min.to(torch.float32))
        self.register_buffer("g_log_max", g_log_max.to(torch.float32))
        self.register_buffer("g_eps", g_eps.to(torch.float32))

        # dt (log10 min/max from cfg.dt_min/cfg.dt_max)
        self.register_buffer("dt_log_min", torch.tensor(float(dt_log_min), dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(float(dt_log_max), dtype=torch.float32))
        self.register_buffer("dt_eps", torch.tensor(float(dt_eps), dtype=torch.float32))

    @property
    def S(self) -> int:
        return int(self.s_log_mean.numel())

    @property
    def G(self) -> int:
        return int(self.g_mean.numel())

    def normalize_dt_seconds(self, dt_seconds: torch.Tensor) -> torch.Tensor:
        """Normalize physical dt (seconds) to [0,1] via log10 + min-max, matching preprocessing.py."""
        # preprocessing.py:
        #   dt_norm = clip((log10(max(dt, eps)) - log10(dt_min)) / (log10(dt_max) - log10(dt_min)), 0, 1)
        dt_f = torch.clamp_min(dt_seconds.to(torch.float32), self.dt_eps)
        log_dt = torch.log10(dt_f)

        # build_baked_normalizer checked dt_log_max > dt_log_min (as float32).
        dt_norm = (log_dt - self.dt_log_min) / (self.dt_log_max - self.dt_log_min)
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        return dt_norm.to(dt_seconds.dtype)

    def normalize_species(self, y_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize species to z-space (model input space).

        All species use log-standard normalization:
            z = (log10(max(y, eps)) - log_mean) / log_std
        Vectorized for inference performance.
        """
        y_f = torch.maximum(y_phys.to(torch.float32), self.s_eps)
        z = (torch.log10(y_f) - self.s_log_mean) / self.s_log_std
        return z.to(y_phys.dtype)

    def denormalize_species(self, y_z: torch.Tensor) -> torch.Tensor:
        """
        Convert z-space species back to physical space.

        Inverse of normalize_species (all species use log-standard):
            y = 10^(z * log_std + log_mean)
        Vectorized for inference performance.
        """
        z_f = y_z.to(torch.float32)
        logx = z_f * self.s_log_std + self.s_log_mean
        y = torch.pow(10.0, logx)
        return y.to(y_z.dtype)

    def normalize_globals(self, g_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize globals to the model's expected space.

        preprocessing.py supports:
          standard | min-max | log-min-max | log-standard
        build_baked_normalizer checked every denominator used here (std >= min_std, max > min).
        """
        g_f = g_phys.to(torch.float32)
        cols: List[torch.Tensor] = []
        for j, m in enumerate(self.global_methods):
            mj = _canonical_method(m)
            xj = g_f[:, j]

            if mj == "standard":
                zj = (xj - self.g_mean[j]) / self.g_std[j]
            elif mj == "min-max":
                zj = (xj - self.g_min[j]) / (self.g_max[j] - self.g_min[j])
            elif mj == "log-min-max":
                zj = (torch.log10(torch.clamp_min(xj, self.g_eps[j])) - self.g_log_min[j]) / (
                    self.g_log_max[j] - self.g_log_min[j]
                )
            elif mj == "log-standard":
                zj = (torch.log10(torch.clamp_min(xj, self.g_eps[j])) - self.g_log_mean[j]) / self.g_log_std[j]
            else:
                raise RuntimeError(f"Unsupported global normalization method: {m}")

            cols.append(zj)

        z = torch.stack(cols, dim=-1)
        return z.to(g_phys.dtype)


def build_baked_normalizer(
    manifest: Mapping[str, Any],
    *,
    species_vars: Sequence[str],
    global_vars: Sequence[str],
) -> BakedNormalizer:
    """Build a BakedNormalizer from normalization.json (species/global stats + dt log range).

    Validates that species use log-standard and globals use a supported method, and that every
    statistic the export uses is sane (NaN fails every check): means and ranges finite, stds
    >= the manifest's min_std, and every range used as a denominator (min-max, log-min-max, dt)
    hi > lo on the float32 values baked. Then
    packs the per-key statistics into float32 buffers for the exported module.
    """
    methods_map = manifest.get("normalization_methods")
    if not isinstance(methods_map, Mapping):
        raise TypeError("normalization.json: expected 'normalization_methods' mapping.")

    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise TypeError("normalization.json: expected 'per_key_stats' mapping.")

    def _positive(key: str) -> float:
        v = float(manifest[key])  # required; canonical preprocessing always writes it
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"normalization.json: {key} must be finite and > 0, got {v!r}")
        return v

    eps_global = _positive("epsilon")
    min_std = _positive("min_std")

    def _finite(k: str, field: str) -> None:
        if not math.isfinite(float(stats[k][field])):
            raise ValueError(f"normalization.json: per_key_stats[{k!r}][{field!r}] must be finite")

    def _std(k: str, field: str) -> None:
        v = float(stats[k][field])
        if not v >= min_std:
            raise ValueError(f"normalization.json: per_key_stats[{k!r}][{field!r}]={v!r} must be >= min_std={min_std}")

    def _range(k: str, lo: str, hi: str) -> None:
        _finite(k, lo)
        _finite(k, hi)
        if not np.float32(stats[k][hi]) > np.float32(stats[k][lo]):
            raise ValueError(f"normalization.json: per_key_stats[{k!r}] needs {hi} > {lo} (as float32)")

    def _method_for(k: str, *, allowed: set[str]) -> str:
        raw = methods_map.get(k)
        m = _canonical_method("" if raw is None else str(raw))
        if m not in allowed:
            raise ValueError(f"Unsupported/missing normalization method for '{k}': {raw!r}")
        return m

    s_methods = [_method_for(k, allowed=_SUPPORTED_SPECIES_METHODS) for k in species_vars]
    g_methods = [_method_for(k, allowed=_SUPPORTED_GLOBAL_METHODS) for k in global_vars]

    # hi > lo only where the range is a denominator: dt, and every global's out-of-range box (linear
    # [min, max] for standard/min-max, [log_min, log_max] for the log methods), which is also the
    # min-max / log-min-max scale. The species log_min/log_max only bound the example inputs (a
    # constant species has log_min == log_max).
    for k in species_vars:  # log-standard
        _finite(k, "log_mean")
        _std(k, "log_std")
        _finite(k, "log_min")
        _finite(k, "log_max")
    for k, m in zip(global_vars, g_methods):
        _range(k, "min", "max")
        if m == "standard":
            _finite(k, "mean")
            _std(k, "std")
        elif m == "log-min-max":
            _range(k, "log_min", "log_max")
        elif m == "log-standard":
            _range(k, "log_min", "log_max")
            _finite(k, "log_mean")
            _std(k, "log_std")

    # Species log stats (required)
    s_log_mean = torch.tensor([float(stats[k]["log_mean"]) for k in species_vars], dtype=torch.float32)
    s_log_std = torch.tensor([float(stats[k]["log_std"]) for k in species_vars], dtype=torch.float32)
    s_log_min = torch.tensor([float(stats[k]["log_min"]) for k in species_vars], dtype=torch.float32)
    s_log_max = torch.tensor([float(stats[k]["log_max"]) for k in species_vars], dtype=torch.float32)
    s_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in species_vars], dtype=torch.float32)

    # Globals stats (required; produced by canonical preprocessing).
    g_mean = torch.tensor([float(stats[k]["mean"]) for k in global_vars], dtype=torch.float32)
    g_std = torch.tensor([float(stats[k]["std"]) for k in global_vars], dtype=torch.float32)
    g_min = torch.tensor([float(stats[k]["min"]) for k in global_vars], dtype=torch.float32)
    g_max = torch.tensor([float(stats[k]["max"]) for k in global_vars], dtype=torch.float32)
    g_log_mean = torch.tensor([float(stats[k]["log_mean"]) for k in global_vars], dtype=torch.float32)
    g_log_std = torch.tensor([float(stats[k]["log_std"]) for k in global_vars], dtype=torch.float32)
    g_log_min = torch.tensor([float(stats[k]["log_min"]) for k in global_vars], dtype=torch.float32)
    g_log_max = torch.tensor([float(stats[k]["log_max"]) for k in global_vars], dtype=torch.float32)
    g_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in global_vars], dtype=torch.float32)

    dt = manifest.get("dt") or {}
    if not isinstance(dt, Mapping) or "log_min" not in dt or "log_max" not in dt:
        raise KeyError("normalization.json: missing dt.log_min / dt.log_max")
    dt_lo, dt_hi = float(dt["log_min"]), float(dt["log_max"])
    if not (math.isfinite(dt_lo) and math.isfinite(dt_hi) and np.float32(dt_hi) > np.float32(dt_lo)):
        raise ValueError(f"normalization.json: dt needs finite log_max > log_min, got {dt_lo!r}, {dt_hi!r}")

    return BakedNormalizer(
        species_vars=species_vars,
        global_vars=global_vars,
        species_methods=s_methods,
        global_methods=g_methods,
        s_log_mean=s_log_mean,
        s_log_std=s_log_std,
        s_log_min=s_log_min,
        s_log_max=s_log_max,
        s_eps=s_eps,
        g_mean=g_mean,
        g_std=g_std,
        g_min=g_min,
        g_max=g_max,
        g_log_mean=g_log_mean,
        g_log_std=g_log_std,
        g_log_min=g_log_min,
        g_log_max=g_log_max,
        g_eps=g_eps,
        dt_log_min=float(dt["log_min"]),
        dt_log_max=float(dt["log_max"]),
        dt_eps=eps_global,
    )


# =============================================================================
# Exportable physical-space 1-step wrapper
# =============================================================================


class OneStepPhysical(nn.Module):
    """
    Wrapper: physical -> (normalize) -> base.forward_step(z) -> (denormalize) -> physical,
    with out-of-range and non-finite rows returned as NaN (module docstring, OOR_TOL).
    """

    def __init__(self, base: nn.Module, norm: BakedNormalizer) -> None:
        super().__init__()
        self.base = base
        self.norm = norm
        # A tensor constant: a Python NaN literal in the graph makes torch.export.save fail (torch 2.8).
        self.register_buffer("nan_fill", torch.tensor(float("nan"), dtype=torch.float32))

        self.S = int(getattr(base, "S"))
        self.G = int(getattr(base, "G"))
        if self.S != norm.S:
            raise RuntimeError(f"S mismatch: model.S={self.S} norm.S={norm.S}")
        if self.G != norm.G:
            raise RuntimeError(f"G mismatch: model.G={self.G} norm.G={norm.G}")

        if not hasattr(base, "forward_step"):
            raise TypeError("Base model must implement forward_step(y_z, dt_norm, g_z)")

    def in_range(self, y_phys: torch.Tensor, dt_seconds: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        """[B] bool: dt and every global inside the train-split box (within OOR_TOL) and y finite.

        Computed from the UNCLAMPED inputs, as fractions of the range in the normalizer's coordinate
        (log10 for dt and the log-* methods, linear otherwise); NaN, +-inf and dt or P <= 0 fail.
        """
        n = self.norm
        cols = [(torch.log10(dt_seconds.to(torch.float32)) - n.dt_log_min) / (n.dt_log_max - n.dt_log_min)]
        g = g_phys.to(torch.float32)
        for j, m in enumerate(n.global_methods):
            if _canonical_method(m) in ("log-min-max", "log-standard"):
                cols.append((torch.log10(g[:, j]) - n.g_log_min[j]) / (n.g_log_max[j] - n.g_log_min[j]))
            else:
                cols.append((g[:, j] - n.g_min[j]) / (n.g_max[j] - n.g_min[j]))
        f = torch.stack(cols, dim=-1)  # [B, 1 + G]
        return ((f >= -OOR_TOL) & (f <= 1.0 + OOR_TOL)).all(dim=-1) & torch.isfinite(y_phys).all(dim=-1)

    def forward(self, y_phys: torch.Tensor, dt_seconds: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        """Physical-space one step: normalize (y, dt, g) -> base.forward_step -> denormalize to y_next_phys.

        The clamps in the normalizer stay (the network never sees a non-finite value); rows that
        fail in_range() are replaced by NaN.
        """
        y_z = self.norm.normalize_species(y_phys)
        dt_norm = self.norm.normalize_dt_seconds(dt_seconds)  # [B]
        g_z = self.norm.normalize_globals(g_phys)
        y_next_z = self.base.forward_step(y_z, dt_norm, g_z)
        y_next = self.norm.denormalize_species(y_next_z)
        ok = self.in_range(y_phys, dt_seconds, g_phys)
        return torch.where(ok[:, None], y_next, self.nan_fill.to(y_next.dtype))


def _make_example_inputs(
    norm: BakedNormalizer,
    *,
    B: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a numerically sane, in-range example batch for export (drawn from `generator`).

    We sample y and dt in *log space* within the min/max ranges captured during preprocessing
    (train split) and g inside the train-split [min, max] box, so every row is in range.
    """
    S = norm.S
    G = norm.G

    # y: sample log10(y) uniformly between observed log_min/log_max.
    u = torch.rand(B, S, device=device, dtype=torch.float32, generator=generator)
    logy = norm.s_log_min.to(device) + u * (norm.s_log_max.to(device) - norm.s_log_min.to(device))
    y = torch.pow(10.0, logy)

    # dt_seconds: sample log10(dt) uniformly between cfg log_min/log_max.
    u_dt = torch.rand(B, device=device, dtype=torch.float32, generator=generator)
    logdt = norm.dt_log_min.to(device) + u_dt * (norm.dt_log_max.to(device) - norm.dt_log_min.to(device))
    dt_seconds = torch.pow(10.0, logdt)

    # g: sample inside the observed min/max box.
    u_g = torch.rand(B, G, device=device, dtype=torch.float32, generator=generator)
    g = norm.g_min.to(device) + u_g * (norm.g_max.to(device) - norm.g_min.to(device))

    return y, dt_seconds, g


def _out_of_range_rows(norm: BakedNormalizer, *, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """A batch whose row 0 is in range and rows 1.. each break one input (dt, P, T or y): every one must be NaN."""
    gen = torch.Generator(device=device).manual_seed(EXPORT_VERIFY_SEED)
    y, dt, g = _make_example_inputs(norm, B=7, device=device, generator=gen)
    dt_lo, dt_hi = float(10.0 ** norm.dt_log_min), float(10.0 ** norm.dt_log_max)
    dt[1] = dt_hi * 10.0          # above the dt range
    dt[2] = dt_lo / 10.0          # below the dt range
    dt[3] = 0.0                   # non-positive dt
    g[4, 0] = norm.g_max[0] * 10.0  # P above the box
    g[5, 1] = norm.g_min[1] * 0.5   # T below the box
    y[6, 0] = float("nan")        # non-finite species input
    return y, dt, g


def _verify_export(module: Any, *, eager: nn.Module, norm: BakedNormalizer, device: torch.device) -> None:
    """Export gate on `device`: fail unless `module` behaves like the eager wrapper.

    In-range inputs (seeded, B in (1, 7, B_MAX)) must give [B, S] outputs that are all finite and
    equal to the eager wrapper's within EXPORT_PARITY_RTOL (atol 0); B_MAX + 1 must be refused; in
    the out-of-range batch exactly the rows that break the box (or are not finite) must be NaN and
    the in-range row must match.
    """
    gen = torch.Generator(device=device).manual_seed(EXPORT_VERIFY_SEED)
    S = norm.S
    for B in (1, 7, B_MAX):
        y, dt, g = _make_example_inputs(norm, B=B, device=device, generator=gen)
        with torch.inference_mode():
            out, ref = module(y, dt, g), eager(y, dt, g)
        if tuple(out.shape) != (B, S):
            raise RuntimeError(f"export gate: input B={B} produced output {tuple(out.shape)}, expected {(B, S)}")
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError(f"export gate: non-finite output for in-range inputs at B={B}")
        torch.testing.assert_close(out, ref, rtol=EXPORT_PARITY_RTOL, atol=0.0)
    y, dt, g = _make_example_inputs(norm, B=B_MAX + 1, device=device, generator=gen)
    try:
        with torch.inference_mode():
            module(y, dt, g)
    except RuntimeError:
        pass  # torch.export's input-constraint check: B above the exported range
    else:
        raise RuntimeError(f"export gate: B={B_MAX + 1} > B_MAX={B_MAX} was not refused")
    y, dt, g = _out_of_range_rows(norm, device=device)
    with torch.inference_mode():
        out, ref = module(y, dt, g), eager(y, dt, g)
    nan_rows = torch.isnan(out).all(dim=-1)
    expected = torch.ones_like(nan_rows)
    expected[0] = False
    if not torch.equal(nan_rows, expected) or not bool(torch.isfinite(out[0]).all()):
        raise RuntimeError(f"export gate: NaN rows {nan_rows.tolist()} != expected {expected.tolist()}")
    torch.testing.assert_close(out[0], ref[0], rtol=EXPORT_PARITY_RTOL, atol=0.0)


def input_out_of_range_rows(y_phys: np.ndarray, dt_seconds: np.ndarray, g_phys: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """[B] bool from the INPUTS and an export's metadata: rows the export returns as NaN.

    Same rule as OneStepPhysical.in_range (float32, metadata oor_box and oor_tol): dt or a global
    outside its box by more than oor_tol of the range, or a non-finite species input.
    """
    tol = np.float32(meta["oor_tol"])
    bad = np.logical_not(np.isfinite(np.asarray(y_phys)).all(axis=1))
    g = np.asarray(g_phys)
    cols = [("dt_seconds", np.asarray(dt_seconds))] + [(name, g[:, j]) for j, name in enumerate(meta["global_variables"])]
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, x in cols:
            box = meta["oor_box"][name]
            lo, hi = np.float32(box["lo"]), np.float32(box["hi"])
            x32 = x.astype(np.float32)
            v = np.log10(x32) if box["coordinate"] == "log10" else x32
            f = (v - lo) / (hi - lo)
            bad |= np.logical_not((f >= -tol) & (f <= np.float32(1.0) + tol))
    return bad


def _strip_stack_traces(ep: torch.export.ExportedProgram) -> None:
    """Drop node stack traces before saving: they carry absolute checkout paths and line numbers
    (a re-export from another checkout would differ), and inference does not need them."""
    for gm in ep.graph_module.modules():
        if isinstance(gm, torch.fx.GraphModule):
            for n in gm.graph.nodes:
                n.meta.pop("stack_trace", None)


# =============================================================================
# Manifest/config resolution
# =============================================================================


def _validate_manifest_vs_config(cfg: Mapping[str, Any], manifest: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Strong correctness check: channel order must match config <-> manifest.

    Config and manifest must both provide matching channel order.
    """
    cfg_data = cfg.get("data")
    if not isinstance(cfg_data, Mapping):
        raise TypeError("config missing data section")
    cfg_species = cfg_data.get("species_variables")
    cfg_globals = cfg_data.get("global_variables")

    man_species = list(manifest.get("species_variables", []) or [])
    man_globals = list(manifest.get("global_variables", []) or [])

    if not isinstance(cfg_species, list) or not all(isinstance(x, str) for x in cfg_species) or not cfg_species:
        raise TypeError("config: bad data.species_variables")
    for source, names in (("config data.species_variables", cfg_species), ("normalization.json species_variables", man_species)):
        dups = sorted({x for x in names if names.count(x) > 1})
        if dups:
            raise ValueError(f"{source}: duplicate species names {dups}")
    if list(cfg_species) != man_species:
        raise ValueError("species_variables mismatch between config and normalization.json (order matters).")
    species = list(cfg_species)

    if not isinstance(cfg_globals, list) or not all(isinstance(x, str) for x in cfg_globals):
        raise TypeError("config: bad data.global_variables")
    if list(cfg_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"config.data.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")
    if list(man_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"normalization.json global_variables must be exactly {list(_REQUIRED_GLOBALS)}")
    if list(cfg_globals) != man_globals:
        raise ValueError("global_variables mismatch between config and normalization.json (order matters).")
    globals_ = list(cfg_globals)

    return species, globals_


def _resolve_processed_dir(cfg: Mapping[str, Any], *, cfg_path: Path) -> Path:
    """
    Locate processed_dir containing normalization.json.

    Strict no-fallback contract: the sole candidate is either
    PROCESSED_DIR_OVERRIDE (when set; for relocated datasets) or
    cfg.paths.processed_dir resolved against the config directory. The candidate
    must contain normalization.json or this hard-fails.
    """
    if PROCESSED_DIR_OVERRIDE is not None:
        configured = PROCESSED_DIR_OVERRIDE
        candidate = _resolve_path_like(PROCESSED_DIR_OVERRIDE, base=REPO_ROOT)
        source = "PROCESSED_DIR_OVERRIDE"
    else:
        paths = cfg.get("paths", {}) or {}
        if not isinstance(paths, Mapping):
            raise TypeError("config missing paths section")

        processed = paths.get("processed_dir")
        if not isinstance(processed, str) or not processed.strip():
            raise KeyError("config.paths.processed_dir is required")

        configured = processed
        candidate = _resolve_path_like(processed, base=cfg_path.parent)
        source = "config.paths.processed_dir"

    if (candidate / "normalization.json").exists():
        return candidate

    raise FileNotFoundError(
        "normalization.json not found under processed_dir (no fallbacks).\n"
        f"Source: {source}\nConfigured: {configured}\nTried:\n  - {candidate}"
    )


def _parse_devices(raw_devices: str) -> List[str]:
    """Parse the comma-separated EXPORT_DEVICES string into a de-duplicated device list."""
    raw = str(raw_devices).strip().lower()
    if not raw:
        raise ValueError("EXPORT_DEVICES produced empty list")
    devs = [d.strip().lower() for d in raw.split(",") if d.strip()]
    if not devs:
        raise ValueError("EXPORT_DEVICES produced empty list")
    seen: set[str] = set()
    out: List[str] = []
    for d in devs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    tags = ["cuda" if d.startswith("cuda") else d for d in out]
    if len(tags) != len(set(tags)):
        raise ValueError("devices list contains duplicate export targets (e.g., both 'cuda' and 'cuda:0')")
    return out


def _resolve_export_devices(devices: Sequence[str]) -> List[str]:
    """Return the requested devices, raising if any of them is unavailable (no skip, no CPU fallback)."""
    for dev in devices:
        if dev.startswith("cuda"):
            if not _cuda_available():
                raise RuntimeError(
                    f"CUDA requested in EXPORT_DEVICES ({dev!r}) but torch.cuda.is_available() is false; "
                    "set EXPORT_DEVICES = 'cpu' to export CPU only"
                )
        elif dev != "cpu":
            raise ValueError(f"Unsupported device '{dev}'. Use cpu|cuda (or cuda:0 style if needed).")
    return list(devices)


def _default_out_for(run_dir: Path, device_tag: str) -> Path:
    """Return the canonical export artifact path: <run_dir>/export_<device>_dynB_1step_phys.pt2."""
    return (run_dir / f"export_{device_tag}_dynB_1step_phys.pt2").resolve()


# =============================================================================
# Export build (shared with testing/aoti_export.py)
# =============================================================================


def _require_manifest_keys(manifest: Mapping[str, Any], path: Path) -> None:
    """The keys the metadata copies must exist in the manifest."""
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise KeyError(
            f"{path} has no {missing} (built before 2026-09-30): rebuild the processed data with "
            "processing/preprocessing.py (the export metadata must state units and the trained domain)"
        )


def _git_or_none(value: Any) -> Any:
    """utils.git_state values, with the 'error: ...' of a non-checkout recorded as null."""
    return None if isinstance(value, str) and value.startswith("error:") else value


def _oor_box(norm: BakedNormalizer) -> Dict[str, Any]:
    """The exact float32 box OneStepPhysical.in_range tests, per input in its own coordinate."""
    box: Dict[str, Any] = {
        "dt_seconds": {"coordinate": "log10", "lo": float(norm.dt_log_min), "hi": float(norm.dt_log_max)}
    }
    for j, (name, m) in enumerate(zip(norm.global_vars, norm.global_methods, strict=True)):
        if _canonical_method(m) in ("log-min-max", "log-standard"):
            box[name] = {"coordinate": "log10", "lo": float(norm.g_log_min[j]), "hi": float(norm.g_log_max[j])}
        else:
            box[name] = {"coordinate": "linear", "lo": float(norm.g_min[j]), "hi": float(norm.g_max[j])}
    return box


def _coverage_summary(coverage: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact (T, P) coverage for the metadata: train-window marginals and the lowest-retention cell."""
    t_marg: Dict[str, int] = {}
    p_marg: Dict[str, int] = {}
    for row in coverage["train_windows"]:
        t_key, p_key = f"{row['T_lo_K']:g}", f"{row['log10P_lo']:g}"
        t_marg[t_key] = t_marg.get(t_key, 0) + int(row["windows"])
        p_marg[p_key] = p_marg.get(p_key, 0) + int(row["windows"])
    return {
        "units": coverage["units"],
        "T_bin_K": coverage["T_bin_K"],
        "log10P_bin_dex": coverage["log10P_bin_dex"],
        "train_windows_by_T_lo_K": t_marg,
        "train_windows_by_log10P_lo": p_marg,
        "lowest_retention_cell": coverage["lowest_retention_cell"],
    }


def build_export(
    run_dir: Path,
    checkpoint: str,
    device: torch.device,
    example_batch: int,
) -> Tuple[torch.export.ExportedProgram, Dict[str, Any], Any]:
    """Build the gated one-step physical-space export for `device` and its metadata.

    The run must be finished (<run_dir>/COMPLETED). Loads <run_dir>/config.resolved.json, the
    checkpoint (EMA weights preferred; its model block must match the config's and its norm_sha256
    the manifest's) and normalization.json, wraps base+normalizer (with the NaN-row guard), exports
    with a dynamic batch dim and runs the export gate on `device`.

    Returns (ep, metadata, verify): verify(module) re-runs the gate against the eager wrapper, for a
    reloaded artifact before it replaces the canonical file.
    """
    if not (run_dir / _COMPLETED_NAME).is_file():
        raise FileNotFoundError(
            f"{run_dir} has no {_COMPLETED_NAME}: the run did not finish (or is being extended); export "
            "only a finished run"
        )
    if example_batch < MIN_EXAMPLE_BATCH:
        raise ValueError(f"example_batch must be >= {MIN_EXAMPLE_BATCH} (torch.export specializes size 1), got {example_batch}")
    cfg, cfg_path = _load_resolved_config(run_dir)
    ckpt_path = _resolve_checkpoint_path(checkpoint, cfg_path=cfg_path)

    processed_dir = _resolve_processed_dir(cfg, cfg_path=cfg_path)
    manifest_path = processed_dir / "normalization.json"
    manifest = _load_json(manifest_path)
    _require_manifest_keys(manifest, manifest_path)
    norm_sha = normalization_sha256(manifest)

    species_vars, global_vars = _validate_manifest_vs_config(cfg, manifest)

    base = create_model(cfg, manifest)
    ckpt_record, model_block = _load_weights_strict(base, ckpt_path, expected_model=cfg["model"], norm_sha256=norm_sha)
    base = _freeze_for_inference(base.to(device=device))

    norm = build_baked_normalizer(manifest, species_vars=species_vars, global_vars=global_vars)
    norm = norm.to(device=device)
    norm.eval()

    step = OneStepPhysical(base, norm).to(device=device)
    step = _freeze_for_inference(step)

    gen = torch.Generator(device=device).manual_seed(EXAMPLE_INPUT_SEED)
    example_inputs = _make_example_inputs(norm, B=int(example_batch), device=device, generator=gen)

    B = Dim("B", min=int(B_MIN), max=int(B_MAX))
    dynamic_shapes = (
        {0: B},  # y_phys: (B, S)
        {0: B},  # dt_seconds: (B,)
        {0: B},  # g_phys: (B, G)
    )

    ep = torch.export.export(step, example_inputs, dynamic_shapes=dynamic_shapes, strict=bool(EXPORT_STRICT))

    def verify(module: Any) -> None:
        _verify_export(module, eager=step, norm=norm, device=device)

    # Gate the in-memory program on the export device (the saved file is gated again on reload).
    verify(ep.module())

    git = git_state(Path(__file__).resolve().parent)
    precision, runtime = cfg["precision"], cfg["runtime"]
    meta = {
        "format": "1step_physical_dynB",
        "metadata_version": METADATA_VERSION,
        "run_dir": _to_repo_relative_str(run_dir),
        "config_path": _to_repo_relative_str(cfg_path),
        "checkpoint_path": _to_repo_relative_str(ckpt_path),
        "checkpoint_sha256": file_sha256(ckpt_path),
        **ckpt_record,
        "training_tf32": bool(precision["allow_tf32"]) and not bool(runtime["deterministic"]),
        "normalization_path": _to_repo_relative_str(manifest_path),
        "normalization_sha256": file_sha256(manifest_path),
        "norm_sha256": norm_sha,
        "export_code_git_commit": _git_or_none(git["head"]),
        "export_code_git_dirty": _git_or_none(git["dirty"]),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "export_device": str(device),
        "export_device_tag": device.type,
        "export_dtype": "float32",
        "model": model_block,
        "species_variables": list(species_vars),
        "global_variables": list(global_vars),
        "units": manifest["units"],
        "dt_validity": DT_VALIDITY,
        "normalization_methods": dict(manifest["normalization_methods"]),
        "per_key_stats": manifest["per_key_stats"],
        "epsilon": float(manifest["epsilon"]),
        "species_floor": "species inputs <= epsilon are floored at epsilon before log10 (not range-checked)",
        "dt_log10_min": float(manifest["dt"]["log_min"]),
        "dt_log10_max": float(manifest["dt"]["log_max"]),
        "dt_min_seconds": float(10.0 ** float(manifest["dt"]["log_min"])),
        "dt_max_seconds": float(10.0 ** float(manifest["dt"]["log_max"])),
        "oor_policy": "nan_row",
        "oor_tol": OOR_TOL,
        "oor_box": _oor_box(norm),
        "window_start_time_s": manifest["window_start_time_s"],
        "coverage": _coverage_summary(manifest["coverage"]),
        "species_sum": manifest.get("species_sum", "not computed"),
        "dynamic_batch": {"min": int(B_MIN), "max": int(B_MAX)},
        "signature": {
            "inputs": {"y_phys": ["B", "S"], "dt_seconds": ["B"], "g_phys": ["B", "G"]},
            "output": {"y_next_phys": ["B", "S"]},
        },
    }
    return ep, meta, verify


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Export the run's checkpoint for each requested device and save it with embedded metadata."""
    _print_device_diag()

    run_dir = resolve_run_dir()
    for dev in _resolve_export_devices(_parse_devices(EXPORT_DEVICES)):
        device = torch.device(dev)
        ep, meta, verify = build_export(run_dir, CHECKPOINT, device, int(EXAMPLE_BATCH))
        _strip_stack_traces(ep)

        out_path = _default_out_for(run_dir, device.type)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        extra = {"metadata.json": json.dumps(meta, indent=2, sort_keys=True)}
        save_artifact(
            out_path,
            lambda staging: torch.export.save(ep, staging, extra_files=extra),  # noqa: B023 - called now
            verify=lambda staging: verify(torch.export.load(str(staging)).module()),  # noqa: B023 - called now
        )
        print(f"Saved export ({device.type}): {out_path}")


if __name__ == "__main__":
    main()
