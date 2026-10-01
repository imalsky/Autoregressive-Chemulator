#!/usr/bin/env python3
"""
utils.py - Small shared utilities.

This project intentionally keeps shared code minimal and explicit.

Included helpers:
  - ensure_dir: create directories idempotently
  - load_json_config: strict JSON load (no duplicate keys, no NaN/Infinity) with helpful errors
  - atomic_write_json: write JSON safely (write temp + atomic rename)
  - comment keys, dotted flattening and the model-block check shared by main.py and export.py
  - PrecisionConfig / parse_precision_config: central dtype configuration for training

Note:
  preprocessing.py is intentionally self-contained and does not import this module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import torch

# Config keys starting with this prefix are comments, allowed at any depth. Any other
# '_'-prefixed key is an unknown key, so a commented-out option cannot silently fall back
# to its default.
COMMENT_KEY_PREFIX = "_comment"


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reject_duplicate_keys(pairs: List[tuple]) -> Dict[str, Any]:
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


def load_json_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load a JSON object strictly: duplicate keys and NaN/Infinity are errors."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(
                f,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite_constant,
            )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {file_path} at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e
    except ValueError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a JSON object at top-level: {file_path}.")
    return obj


def is_comment_key(key: Any) -> bool:
    """True for config comment keys (see COMMENT_KEY_PREFIX)."""
    return isinstance(key, str) and key.startswith(COMMENT_KEY_PREFIX)


def strip_comment_keys(obj: Any) -> Any:
    """Return a copy of a JSON-like value with comment keys removed at every depth."""
    if isinstance(obj, Mapping):
        return {k: strip_comment_keys(v) for k, v in obj.items() if not is_comment_key(k)}
    if isinstance(obj, list):
        return [strip_comment_keys(v) for v in obj]
    return obj


def flatten_dotted(obj: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings to {"a.b.c": leaf}; lists are leaves."""
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dotted(v, path))
        else:
            out[path] = v
    return out


def dotted_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    """Sorted dotted keys whose values differ between two nested mappings (comment keys ignored)."""
    fa = flatten_dotted(strip_comment_keys(a))
    fb = flatten_dotted(strip_comment_keys(b))
    missing = object()
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k, missing) != fb.get(k, missing))


def require_same_model_block(saved: Any, expected: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Raise unless a checkpoint's saved model block equals the config's (comment keys ignored).

    Shape mismatches already fail in load_state_dict; this also catches the knobs that do not
    change parameter shapes (predict_delta, activation, layer_norm_eps, dropout, ...).
    Returns the saved block without comment keys.
    """
    if not isinstance(saved, Mapping):
        raise KeyError(f"{source}: checkpoint has no hyper_parameters['model'] block to check against the config")
    diff = dotted_differences(saved, expected)
    if diff:
        raise ValueError(
            f"{source}: checkpoint model hyperparameters differ from the config at "
            + ", ".join(f"model.{k}" for k in diff)
        )
    return strip_comment_keys(saved)


def non_finite_tensor_keys(state: Mapping[str, Any]) -> List[str]:
    """Keys of floating tensors in a state dict that hold NaN or Inf."""
    return [
        k for k, v in state.items()
        if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())
    ]


def _fsync_dir_best_effort(directory: Path) -> None:
    """Best-effort fsync of a directory entry (helps durability after rename)."""
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(directory), flags)
    except Exception:
        return

    try:
        os.fsync(dir_fd)
    except Exception:
        pass
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass


def atomic_write_json(path: Union[str, os.PathLike], obj: Any, *, indent: int = 2) -> None:
    """Atomically write JSON to disk using write-to-temp-then-rename."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=int(indent), sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        # Atomic on POSIX when src/dst are on same filesystem.
        os.replace(tmp, out_path)
        _fsync_dir_best_effort(out_path.parent)

    finally:
        # If anything failed before os.replace, clean up the temp file.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


# ==============================================================================
# Precision (centralized dtype configuration)
# ==============================================================================

_ALLOWED_DTYPES: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_torch_dtype(value: Any, *, key: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype.

    Supported (case-insensitive):
      - bfloat16
      - float32
      - float64

    Note:
      float16 is intentionally unsupported in this codebase.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"bad type: {key} (expected non-empty string)")
    name = value.strip().lower()
    if name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"unsupported dtype for {key}: '{value}' (supported: bfloat16, float32, float64)"
        )
    return _ALLOWED_DTYPES[name]


def require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing: {key}")
    return mapping[key]


def require_dict(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    val = require(mapping, key)
    if not isinstance(val, dict):
        raise TypeError(f"bad type: {key}")
    return val


def require_dotted(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    cur: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"missing: {dotted_key}")
        cur = cur[part]
    return cur


def as_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"bad type: {key}")
    return value.strip()


def as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"bad type: {key}")
    return int(value)


def as_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"bad type: {key}")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return float(value)


def as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"bad type: {key}")
    return bool(value)


@dataclass(frozen=True)
class PrecisionConfig:
    """Resolved precision configuration (torch dtypes + Lightning precision string).

    The preloaded dataset is always float32 (the shard dtype); there is no storage-dtype key.
    """

    compute_dtype: torch.dtype
    amp_mode: str  # "mixed" | "true"
    model_dtype: torch.dtype
    input_dtype: torch.dtype
    loss_dtype: torch.dtype

    @property
    def lightning_precision(self) -> str:
        """Lightning precision string derived from compute_dtype + amp_mode."""
        if self.compute_dtype == torch.bfloat16:
            return "bf16-mixed" if self.amp_mode == "mixed" else "bf16-true"
        if self.compute_dtype == torch.float32:
            return "32-true"
        if self.compute_dtype == torch.float64:
            return "64-true"
        raise ValueError(f"unsupported compute_dtype: {self.compute_dtype}")

def parse_precision_config(cfg: Mapping[str, Any]) -> PrecisionConfig:
    """Parse and validate cfg["precision"] into a PrecisionConfig."""
    pcfg_raw = cfg.get("precision", None)
    if not isinstance(pcfg_raw, Mapping):
        raise KeyError("missing/invalid: precision (expected object)")

    compute_dtype = parse_torch_dtype(require(pcfg_raw, "compute_dtype"), key="precision.compute_dtype")
    amp_mode = as_str(require(pcfg_raw, "amp_mode"), "precision.amp_mode").lower()
    if amp_mode not in ("mixed", "true"):
        raise ValueError("precision.amp_mode must be 'mixed' or 'true'")

    model_dtype = parse_torch_dtype(require(pcfg_raw, "model_dtype"), key="precision.model_dtype")
    input_dtype = parse_torch_dtype(require(pcfg_raw, "input_dtype"), key="precision.input_dtype")
    loss_dtype = parse_torch_dtype(require(pcfg_raw, "loss_dtype"), key="precision.loss_dtype")

    # Cross-validation: avoid silent dtype mismatches.
    if compute_dtype in (torch.float32, torch.float64):
        if amp_mode != "true":
            raise ValueError("precision.amp_mode must be 'true' when compute_dtype is float32/float64")
        if model_dtype != compute_dtype:
            raise ValueError("precision.model_dtype must equal precision.compute_dtype for float32/float64 compute")
        if input_dtype != compute_dtype:
            raise ValueError("precision.input_dtype must equal precision.compute_dtype for float32/float64 compute")

    if compute_dtype == torch.bfloat16 and amp_mode == "true":
        # Pure bf16: model + inputs must be bf16 to avoid dtype mismatches.
        if model_dtype != torch.bfloat16:
            raise ValueError("precision.model_dtype must be bfloat16 when compute_dtype=bfloat16 and amp_mode=true")
        if input_dtype != torch.bfloat16:
            raise ValueError("precision.input_dtype must be bfloat16 when compute_dtype=bfloat16 and amp_mode=true")

    if amp_mode == "mixed":
        # Mixed precision keeps fp32 master weights and fp32 inputs; autocast lowers only the matmuls.
        # bf16 weights would lose the small AdamW updates, and bf16 inputs quantize the z-space state
        # (bf16 storage perturbed states by a median 3.1e-3 dex in the audit, above the ~7e-4 target).
        if model_dtype != torch.float32:
            raise ValueError("precision.model_dtype must be float32 when amp_mode=mixed (fp32 master weights)")
        if input_dtype != torch.float32:
            raise ValueError("precision.input_dtype must be float32 when amp_mode=mixed (bf16 inputs quantize the state)")
        predict_delta = as_bool(require(require(cfg, "model"), "predict_delta"), "model.predict_delta")
        if not predict_delta:
            raise ValueError(
                "precision.amp_mode=mixed requires model.predict_delta=true: without the residual add the "
                "whole next state comes out of the bf16 trunk, which carries the rollout state at bf16 "
                "resolution and erases most per-step changes"
            )

    # The loss is accumulated in at least float32: a bf16 loss rounds the per-step log10 errors
    # the training target is measured in.
    if loss_dtype not in (torch.float32, torch.float64):
        raise ValueError("precision.loss_dtype must be float32 or float64 (a bf16 loss rounds the per-step errors)")

    return PrecisionConfig(
        compute_dtype=compute_dtype,
        amp_mode=amp_mode,
        model_dtype=model_dtype,
        input_dtype=input_dtype,
        loss_dtype=loss_dtype,
    )
