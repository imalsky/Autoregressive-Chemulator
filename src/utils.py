#!/usr/bin/env python3
"""
utils.py - Small shared utilities.

This project intentionally keeps shared code minimal and explicit.

Included helpers:
  - ensure_dir: create directories idempotently
  - load_json_config: strict JSON load (no duplicate keys, no NaN/Infinity) with helpful errors
  - atomic_write_json: write JSON safely (write temp + atomic rename)
  - comment keys, dotted flattening and the model-block check shared by main.py and export.py
  - normalization_sha256: the fingerprint that ties a checkpoint or export to its normalization
  - file_sha256, git_state, collect_provenance: best-effort run and export provenance
  - lock_processed_dir_shared: a reader's shared flock on the processed data (vs preprocessing)
  - PrecisionConfig / parse_precision_config: central dtype configuration for training

Note:
  preprocessing.py is intentionally self-contained and does not import this module.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch

# Config keys starting with this prefix are comments, allowed at any depth. Any other
# '_'-prefixed key is an unknown key, so a commented-out option cannot silently fall back
# to its default.
COMMENT_KEY_PREFIX = "_comment"

# The normalization.json keys that define the z-space a model is trained in (the transforms and
# their statistics). Their canonical-JSON sha256 is the normalization fingerprint; additive
# descriptive manifest keys (units, coverage, ...) do not change it.
NORM_FINGERPRINT_KEYS = (
    "normalization_methods",
    "per_key_stats",
    "epsilon",
    "min_std",
    "dt",
    "species_variables",
    "global_variables",
)
# Streamed hashing of checkpoints / artifacts (bytes per read).
_HASH_CHUNK_BYTES = 1 << 20
# processing/preprocessing.py flocks <processed_dir>.lock (a persistent 0-byte sibling of the resolved
# processed_dir; must equal its _LOCK_SUFFIX) exclusively for its whole run. Readers that load the
# manifest and then the shards hold it shared meanwhile, so neither side sees the other half-way.
PROCESSED_DIR_LOCK_SUFFIX = ".lock"
# Mode of a lock file a reader creates: open()'s default, as preprocessing's open(lock_path, "w").
_LOCK_FILE_MODE = 0o666
# A provenance subprocess (git, nvidia-smi) that takes longer than this is recorded as an error.
_PROVENANCE_SUBPROCESS_TIMEOUT_S = 10.0


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
    change parameter shapes (activation, layer_norm_eps, dropout, ...).
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


def normalization_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON (sorted keys, compact) of the manifest's NORM_FINGERPRINT_KEYS."""
    missing = [k for k in NORM_FINGERPRINT_KEYS if k not in manifest]
    if missing:
        raise KeyError(f"normalization manifest missing {missing}")
    sub = {k: manifest[k] for k in NORM_FINGERPRINT_KEYS}
    blob = json.dumps(sub, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, os.PathLike]) -> str:
    """Streamed sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_processed_dir_shared(processed_dir: Union[str, os.PathLike]) -> int:
    """Take a shared, non-blocking flock on <processed_dir>.lock; return its fd (os.close releases it).

    Fails at once while a preprocessing run holds the lock (it is rebuilding processed_dir). A
    missing lock file (a build copied without it) is created, as preprocessing creates it, so both
    sides always lock the same file.
    """
    resolved = Path(processed_dir).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"processed_dir {resolved} does not exist (paths.processed_dir)")
    lock_path = resolved.parent / (resolved.name + PROCESSED_DIR_LOCK_SUFFIX)
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, _LOCK_FILE_MODE)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(
            f"{lock_path} is held by a preprocessing run that is rebuilding {resolved}; wait for it to finish"
        ) from None
    return fd


def best_effort(fn: Callable[[], Any]) -> Any:
    """fn(), or an 'error: ...' string: provenance must never stop a run."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        return f"error: {type(e).__name__}: {e}"


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> bytes:
    """stdout of a read-only subprocess (raises on failure or after the provenance timeout)."""
    return subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, timeout=_PROVENANCE_SUBPROCESS_TIMEOUT_S
    ).stdout


def git_state(code_dir: Path) -> Dict[str, Any]:
    """Best-effort git HEAD, dirty flag and sha256 of `git diff HEAD` for the checkout holding code_dir.

    Untracked files do not count as dirty (--untracked-files=no). --no-optional-locks keeps the
    status read-only (no index.lock). Outside a checkout every field is an error string.
    """
    def head() -> str:
        return _run(["git", "-C", str(code_dir), "rev-parse", "HEAD"]).decode().strip()

    def dirty() -> bool:
        out = _run(["git", "-C", str(code_dir), "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"])
        return bool(out.strip())

    def diff_sha() -> str:
        return hashlib.sha256(_run(["git", "-C", str(code_dir), "--no-optional-locks", "diff", "HEAD"])).hexdigest()

    return {"head": best_effort(head), "dirty": best_effort(dirty), "diff_sha256": best_effort(diff_sha)}


def _package_versions() -> Dict[str, Any]:
    """Installed versions of the packages training depends on."""
    return {
        name: best_effort(lambda name=name: importlib.metadata.version(name))
        for name in ("torch", "lightning", "numpy")
    }


def _gpu_info() -> Any:
    """Name, UUID and driver of the visible CUDA device 0 (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    driver = best_effort(lambda: _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id=GPU-{uuid}"]
    ).decode().strip())
    return {"name": props.name, "uuid": f"GPU-{uuid}", "driver": driver}


def collect_provenance(code_dir: Path, **fields: Any) -> Dict[str, Any]:
    """One best-effort environment record (every field an error string on failure), plus `fields`.

    Records time, git state of code_dir's checkout, python/package versions, CUDA/cuDNN, the GPU,
    host, CUDA_VISIBLE_DEVICES, CPU affinity, thread settings and the effective float32 matmul
    precision of this process.
    """
    rec: Dict[str, Any] = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": git_state(code_dir),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "torch_cuda": torch.version.cuda,
        "cudnn": best_effort(lambda: torch.backends.cudnn.version()),
        "gpu": best_effort(_gpu_info),
        "hostname": best_effort(socket.gethostname),
        "platform_machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": best_effort(lambda: sorted(os.sched_getaffinity(0))),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    rec.update(fields)
    return rec


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
    # The relaxation head runs its network in float32 (forward_step casts its inputs), so the weights must
    # stay float32: 64-true and bf16-true would cast them and fail at the first forward.
    if compute_dtype == torch.float64 or (compute_dtype == torch.bfloat16 and amp_mode == "true"):
        raise ValueError(
            "precision.compute_dtype must be float32, or bfloat16 with amp_mode=mixed "
            "(the relaxation head runs its network in float32)"
        )
    if compute_dtype in (torch.float32, torch.float64):
        if amp_mode != "true":
            raise ValueError("precision.amp_mode must be 'true' when compute_dtype is float32/float64")
        if model_dtype != compute_dtype:
            raise ValueError("precision.model_dtype must equal precision.compute_dtype for float32/float64 compute")
        # float64 inputs carry the rollout state in float64 (forward_step runs the network in float32 and adds
        # the increment in the state's dtype); the shards stay float32.
        if input_dtype not in (compute_dtype, torch.float64):
            raise ValueError(
                "precision.input_dtype must equal precision.compute_dtype or be float64 for float32/float64 compute"
            )

    if amp_mode == "mixed":
        # Mixed precision keeps fp32 master weights and fp32 inputs; autocast lowers only the matmuls.
        # bf16 weights would lose the small AdamW updates, and bf16 inputs quantize the z-space state
        # (bf16 storage perturbed states by a median 3.1e-3 dex).
        if model_dtype != torch.float32:
            raise ValueError("precision.model_dtype must be float32 when amp_mode=mixed (fp32 master weights)")
        if input_dtype != torch.float32:
            raise ValueError("precision.input_dtype must be float32 when amp_mode=mixed (bf16 inputs quantize the state)")

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
