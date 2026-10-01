#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and validate a VULCAN-oriented CPU AOTInductor package.

This script is validation-first:
- it builds one CPU AOTInductor package from the run's best checkpoint (the run directory is
  paths.work_dir of AUTOCHEM_CONFIG_PATH's config, testing/run_dir.py; it must hold COMPLETED)
- it validates that package against a fresh raw export on real VULCAN column inputs (pass/fail) and
  compares the installed raw VULCAN model with the fresh export (informational)
- it checks the host input guard (_check_step_inputs) on valid and invalid inputs
- it benchmarks the direct model-call latency for the raw and AOTI paths
- it writes a JSON report plus a markdown report into the run directory

Pass/fail: in every validation case (each benchmark batch size and dynamic_batch.max at dt_min,
HOST_DT_SECONDS and dt_max from the export metadata, plus one batch with an out-of-range dt row;
model inputs floored at the export's epsilon, the training floor) the AOTI package and the fresh
export must return identical NaN rows, equal to the rows the inputs put outside the trained box; no
+-inf; and max |log10(aoti) - log10(fresh)| <= MATCH_MAX_DLOG10 over the in-range rows (a case whose
rows are all out of range checks the NaN rows only); the cases together must hold at least one
in-range row. Every invalid input must be refused by the guard and B=1 accepted. The package is
compiled under <run_dir>/.partial/ and renamed into place only on a pass; on a failure both reports
are still written, the previous package stays, and the script exits 1.

The package is NATIVE CODE FOR THE BUILD CPU: inductor compiles it with -march=native (AVX-512 and AMX
on daw), so build it on the machine that will run it, or on one with the same CPU generation, OS/glibc
and torch build. The build target is recorded in the package metadata (build_target).

The package does NO input validation (the raw .pt2's inputs are checked by torch.export; the AOTI
package's compiled checks are off unless AOTI_RUNTIME_CHECK_INPUTS is set, which is not recommended:
its compiled lower bound rejects valid B=1 calls and it misses dt/g batch sizes that differ from y's).
Host contract (metadata host_input_contract): CPU float32 C-contiguous y [B, S], dt [B], g [B, G],
one B for all three, B within metadata dynamic_batch. Call the package through _check_step_inputs
(copied into the host; about 2-3 us per call) or convert each input with
np.ascontiguousarray(x, dtype=np.float32) and apply the same shape and B checks.

Metadata: the export metadata plus aoti_example_batch, build_target, build_host and
host_input_contract, embedded twice: runtime-visible through the package loader's get_metadata()
(inductor's aot_inductor.metadata; every value is a JSON-encoded string) and as extra/metadata.json
in the archive.

Implemented, not run on daw: torch 2.8 extracts AOTI packages into a hard-coded directory under the
shared system temp dir, which the daw rules forbid. Run:
  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/aoti_export.py

It does not overwrite ``Emulator/model/model.pt2``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import platform
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import _inductor


REPO_ROOT = Path(__file__).resolve().parents[1]
# testing/export.py supplies the run resolver, CHECKPOINT and the shared build_export().
EXPORT_IMPL_PATH = (Path(__file__).resolve().parent / "export.py").resolve()
# External, machine-specific inputs (edit here): this script needs a VULCAN working
# tree providing a saved column state pickle and the installed raw Emulator model.
# Neither lives in this repo; the script hard-fails with a clear message when absent.
REFERENCE_VULCAN_STATE_PATH = (
    REPO_ROOT.parent / "Misc" / "VULCAN-0D" / "output" / "HD189_short_thermo.vul"
).resolve()
INSTALLED_RAW_MODEL_PATH = (
    REPO_ROOT.parent / "Misc" / "VULCAN-0D" / "Emulator" / "model" / "model.pt2"
).resolve()
# Outputs, written into the resolved run directory.
OUTPUT_AOTI_PACKAGE_NAME = "export_cpu_dynB_1step_phys_aoti.pt2"
OUTPUT_REPORT_NAME = "export_cpu_dynB_1step_phys_aoti_validation.md"
OUTPUT_JSON_NAME = "export_cpu_dynB_1step_phys_aoti_validation.json"
AOTI_EXAMPLE_BATCH: int | None = None  # None => the reference column's nz (floored at MIN_EXAMPLE_BATCH)
BENCHMARK_BATCH_SIZES: tuple[int | str, ...] = (1, "nz", "2*nz")
# The deployment-like validation dt (inherited: Canoe's coupling dt, docs/DECISIONS.md); dt_min and
# dt_max come from the export metadata.
HOST_DT_SECONDS = 6.4
# The out-of-range validation row: one decade above dt_max, unambiguously outside the box.
OOR_PROBE_DT_FACTOR = 10.0
# Floor of a column row's total number density before dividing by it: a guard against an empty row,
# not a physics value (inherited 1e-30; unrelated to the normalization epsilon, which floors the
# model inputs).
TOTAL_DENSITY_FLOOR = 1.0e-30
# Latency benchmark: untimed warm-up calls (first-call allocations, thread-pool start-up), then timed
# calls averaged per (model, batch). Inherited; no recorded rationale.
BENCHMARK_WARMUP_ITERS = 3
BENCHMARK_ITERS = 20
# AOTI-vs-fresh tolerance in dex: about 10x the float32 AOTI-vs-eager noise measured on a 10x2048
# random-weight proxy (9.5e-6 dex) and far below any physically meaningful error; re-check on the
# first trained model.
MATCH_MAX_DLOG10 = 1.0e-4
# Build-target record: the compiler inductor's -march=native resolves through, and the /proc/cpuinfo
# flag prefixes that matter for the generated kernels.
_TARGET_COMPILER = "gcc"
_CPU_FLAG_PREFIXES = ("avx512", "amx")
_CPUINFO_PATH = Path("/proc/cpuinfo")
_SUBPROCESS_TIMEOUT_S = 10.0
HOST_INPUT_CONTRACT = (
    "The AOTI package does no input validation. Inputs: CPU float32 C-contiguous y_phys [B, S], "
    "dt_seconds [B], g_phys [B, G] (S, G from species_variables / global_variables), the same B for all "
    "three, dynamic_batch.min <= B <= dynamic_batch.max. Call through testing/aoti_export.py "
    "_check_step_inputs (copied into the host) or convert with np.ascontiguousarray(x, dtype=np.float32) "
    "and apply the same shape and B checks. Out-of-range dt/P/T rows return NaN (oor_policy)."
)


@dataclass(frozen=True)
class ExportBuildContext:
    """Fresh raw export objects created from the current export code path."""

    exported_program: torch.export.ExportedProgram
    fresh_raw_model: torch.nn.Module
    metadata: dict[str, Any]
    model_species: tuple[str, ...]
    global_variables: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceColumn:
    """One VULCAN column state used to build validation inputs."""

    path: Path
    species: tuple[str, ...]
    y_full: np.ndarray  # shape: (nz, nspecies)
    pco_barye: np.ndarray  # shape: (nz,)
    Tco_K: np.ndarray  # shape: (nz,)
    nz: int


@dataclass(frozen=True)
class ModelCallCase:
    """One prepared model-call case at a specific batch size and dt."""

    batch_size: int
    dt_label: str
    y_phys: torch.Tensor  # shape: (B, S)
    dt_seconds: torch.Tensor  # shape: (B,)
    g_phys: torch.Tensor  # shape: (B, 2)


@dataclass(frozen=True)
class ValidationMetrics:
    """One reference-vs-candidate comparison: NaN-row agreement and the log10 difference."""

    comparison_name: str
    batch_size: int
    dt_label: str
    n_rows: int
    n_nan_rows_reference: int
    n_nan_rows_candidate: int
    nan_masks_equal: bool          # identical NaN entries in both outputs
    nan_rows_match_inputs: bool    # reference NaN rows == the inputs' out-of-range rows
    any_inf: bool
    max_dlog10: float              # over the rows finite in the reference (nan if there are none)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Direct model-call wall-clock timing for one model and batch size."""

    model_label: str
    batch_size: int
    seconds_per_call: float


class _ValidationFailed(Exception):
    """The AOTI package did not match the fresh export (or the input guard failed)."""


def _load_export_impl() -> Any:
    """Load the existing raw-export script as an importable helper module."""

    spec = importlib.util.spec_from_file_location("autochem_export_impl", EXPORT_IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load export helper module from {EXPORT_IMPL_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_export_suffix(name: str) -> str:
    """Drop the training/export ``_evolution`` suffix from species names."""

    suffix = "_evolution"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _load_raw_export(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load one raw ``torch.export.save(...)`` artifact: (callable module, embedded metadata)."""

    if not path.exists():
        raise FileNotFoundError(f"Missing raw export artifact: {path}")

    extra_files = {"metadata.json": ""}
    exported_program = torch.export.load(str(path), extra_files=extra_files)
    if not extra_files["metadata.json"]:
        raise FileNotFoundError(f"Embedded metadata.json is missing from {path}.")
    metadata = json.loads(extra_files["metadata.json"])
    if not isinstance(metadata, dict):
        raise TypeError(f"Embedded metadata must be a JSON object in {path}.")
    return exported_program.module(), metadata


def _load_reference_column(path: Path) -> ReferenceColumn:
    """Load the default VULCAN column state used for real-input validation."""

    if not path.exists():
        raise FileNotFoundError(
            "Reference VULCAN state is required for this script and is missing: "
            f"{path}\n"
            "This is an external, machine-specific input from a VULCAN working tree; "
            "point REFERENCE_VULCAN_STATE_PATH at a local .vul state pickle."
        )

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    variable = payload["variable"]
    atmosphere = payload["atm"]
    y_full = np.asarray(variable["y"], dtype=np.float64)
    pco_barye = np.asarray(atmosphere["pco"], dtype=np.float64)
    Tco_K = np.asarray(atmosphere["Tco"], dtype=np.float64)
    species = tuple(str(species_name) for species_name in variable["species"])

    if y_full.ndim != 2:
        raise ValueError(f"Expected y with shape [nz, nspecies], got {y_full.shape}.")
    nz = int(y_full.shape[0])
    if pco_barye.shape != (nz,):
        raise ValueError(f"Pressure profile must have shape ({nz},), got {pco_barye.shape}.")
    if Tco_K.shape != (nz,):
        raise ValueError(
            f"Temperature profile must have shape ({nz},), got {Tco_K.shape}."
        )

    return ReferenceColumn(
        path=path,
        species=species,
        y_full=y_full,
        pco_barye=pco_barye,
        Tco_K=Tco_K,
        nz=nz,
    )


def _resolve_example_batch(reference_column: ReferenceColumn, *, min_example_batch: int) -> int:
    """The AOTI example batch (the size inductor tunes for): nz by default, floored at min_example_batch."""

    raw = reference_column.nz if AOTI_EXAMPLE_BATCH is None else int(AOTI_EXAMPLE_BATCH)
    return max(int(min_example_batch), raw)


def _resolve_benchmark_batch_sizes(reference_column: ReferenceColumn) -> tuple[int, ...]:
    """Resolve the configured batch-size rules against the reference ``nz``."""

    resolved: list[int] = []
    for value in BENCHMARK_BATCH_SIZES:
        if value == "nz":
            candidate = reference_column.nz
        elif value == "2*nz":
            candidate = 2 * reference_column.nz
        else:
            candidate = int(value)
        if candidate <= 0:
            raise ValueError(f"Benchmark batch sizes must be positive, got {candidate}.")
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _run_text(cmd: list[str]) -> str:
    """stdout of a read-only command, or an 'error: ...' string (build provenance must not fail the run)."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {type(e).__name__}: {e}"


def _cpuinfo_field(text: str, field: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(field) and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _build_target() -> dict[str, Any]:
    """What the package's native code was compiled for (it targets -march=native of this CPU)."""
    help_text = _run_text([_TARGET_COMPILER, "-march=native", "-Q", "--help=target"])
    march = next(
        (ln.split()[-1] for ln in help_text.splitlines() if ln.strip().startswith("-march=") and len(ln.split()) > 1),
        help_text if help_text.startswith("error:") else None,
    )
    cpuinfo = _CPUINFO_PATH.read_text(encoding="utf-8", errors="replace") if _CPUINFO_PATH.exists() else ""
    flags = (_cpuinfo_field(cpuinfo, "flags") or "").split()
    return {
        "platform_machine": platform.machine(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "march_native_resolves_to": march,
        "cpu_flags": sorted(f for f in flags if f.startswith(_CPU_FLAG_PREFIXES)),
        "compiler_version": _run_text([_TARGET_COMPILER, "--version"]).splitlines()[0:1],
        "torch_version": torch.__version__,
    }


def _build_host() -> dict[str, Any]:
    """Build-time thread settings and CPU (the benchmark numbers depend on them)."""
    cpuinfo = _CPUINFO_PATH.read_text(encoding="utf-8", errors="replace") if _CPUINFO_PATH.exists() else ""
    return {
        "torch_num_threads": torch.get_num_threads(),
        "cpu_affinity_size": len(os.sched_getaffinity(0)),
        "cpu_model": _cpuinfo_field(cpuinfo, "model name"),
    }


def _build_fresh_raw_export(
    export_impl: Any,
    *,
    run_dir: Path,
    example_batch: int,
) -> ExportBuildContext:
    """Build the fresh raw CPU export in memory with export.py's shared build_export()."""

    exported_program, metadata, _ = export_impl.build_export(
        run_dir,
        export_impl.CHECKPOINT,
        torch.device("cpu"),
        example_batch,
    )
    metadata["aoti_example_batch"] = int(example_batch)
    metadata["build_target"] = _build_target()
    metadata["build_host"] = _build_host()
    metadata["host_input_contract"] = HOST_INPUT_CONTRACT

    return ExportBuildContext(
        exported_program=exported_program,
        fresh_raw_model=exported_program.module(),
        metadata=metadata,
        model_species=tuple(_strip_export_suffix(name) for name in metadata["species_variables"]),
        global_variables=tuple(str(name) for name in metadata["global_variables"]),
    )


def _compile_aoti_package(
    exported_program: torch.export.ExportedProgram,
    *,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    """Compile the AOTI package to `path` with runtime-visible metadata, and append it to the archive.

    aot_inductor.metadata gets a fresh dict per call (codecache adds AOTI_DEVICE_KEY to the dict it
    is given); every value is JSON-encoded. The extra/metadata.json side entry is kept for readers
    of the archive.
    """
    runtime_metadata = {key: json.dumps(value, sort_keys=True) for key, value in metadata.items()}
    compiled_path = Path(
        _inductor.aoti_compile_and_package(
            exported_program,
            package_path=str(path),
            inductor_configs={"aot_inductor.metadata": runtime_metadata},
        )
    ).resolve()
    if compiled_path != path.resolve():
        raise RuntimeError(f"AOTI package written to {compiled_path}, expected {path}")
    with zipfile.ZipFile(compiled_path, "a") as archive:
        archive_names = archive.namelist()
        if not archive_names:
            raise RuntimeError(f"AOTI package archive is empty: {compiled_path}")
        archive_prefix = ""
        if not any(name.startswith("data/aotinductor/model/") for name in archive_names):
            first_nested_name = next(
                (name for name in archive_names if "/" in name),
                "",
            )
            if first_nested_name:
                archive_prefix = first_nested_name.split("/", 1)[0] + "/"
        archive.writestr(
            f"{archive_prefix}extra/metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True),
        )


def _column_inputs(
    reference_column: ReferenceColumn,
    *,
    model_species: tuple[str, ...],
    floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The column's model inputs (float32, floored at `floor`) and the tracked-species sum / H2 share per row.

    `floor` is the export's epsilon, the floor the model applies itself, so trace values down to the
    training floor reach the comparison unchanged. The sum and the share are taken before the
    float32 cast and the floor (diagnostic only; the pass/fail rule does not use them).
    """
    species_index = {
        species_name: position for position, species_name in enumerate(reference_column.species)
    }
    active_indices = np.asarray(
        [species_index[species_name] for species_name in model_species],
        dtype=int,
    )

    total_density = np.sum(reference_column.y_full, axis=1, keepdims=True)
    y_active = reference_column.y_full[:, active_indices]
    ymix_active = y_active / np.maximum(total_density, TOTAL_DENSITY_FLOOR)
    row_sum = ymix_active.sum(axis=1)
    h2_share = (ymix_active[:, model_species.index("H2")] / row_sum) if "H2" in model_species else None
    composition = {
        "definition": "per column row: sum of the tracked species' mixing ratios (host convention: "
                      "y_i / total density over all host species) and H2's share of that sum",
        "tracked_sum": {"min": float(row_sum.min()), "median": float(np.median(row_sum)), "max": float(row_sum.max()),
                        "per_row": [float(v) for v in row_sum]},
        "h2_share_of_sum": None if h2_share is None else {
            "min": float(h2_share.min()), "median": float(np.median(h2_share)), "max": float(h2_share.max()),
            "per_row": [float(v) for v in h2_share]},
    }
    ymix_sanitized = np.maximum(
        np.clip(ymix_active.astype(np.float32), 0.0, None),
        np.float32(floor),
    )  # shape: (nz, n_model_species)
    pco_barye = reference_column.pco_barye.astype(np.float32)
    Tco_K = reference_column.Tco_K.astype(np.float32)
    g_full = np.column_stack((pco_barye, Tco_K)).astype(np.float32)  # shape: (nz, 2)
    return ymix_sanitized, g_full, composition


def _build_model_call_cases(
    reference_column: ReferenceColumn,
    *,
    y_rows: np.ndarray,
    g_rows: np.ndarray,
    batch_sizes: tuple[int, ...],
    dts: dict[str, float],
) -> list[ModelCallCase]:
    """Real VULCAN model-call cases (column rows tiled to B) for every batch size and dt in `dts`."""

    cases: list[ModelCallCase] = []
    for batch_size in batch_sizes:
        repeat_count = int(math.ceil(float(batch_size) / float(reference_column.nz)))
        y_case = np.concatenate([y_rows] * repeat_count, axis=0)[:batch_size].copy()
        g_case = np.concatenate([g_rows] * repeat_count, axis=0)[:batch_size].copy()
        for label, dt_value in dts.items():
            dt_case = np.full((batch_size,), dt_value, dtype=np.float32)
            cases.append(
                ModelCallCase(
                    batch_size=int(batch_size),
                    dt_label=label,
                    y_phys=torch.from_numpy(y_case.copy()),
                    dt_seconds=torch.from_numpy(dt_case),
                    g_phys=torch.from_numpy(g_case.copy()),
                )
            )
    return cases


def _check_step_inputs(
    y: Any, dt: Any, g: Any, *, S: int, G: int, b_min: int, b_max: int
) -> None:
    """Host input guard for the AOTI package (it validates nothing itself): raise unless the call is valid.

    y [B, S], dt [B], g [B, G]: CPU float32 torch tensors, each contiguous, one shared B with
    b_min <= B <= b_max.
    """
    for name, t, ndim in (("y", y, 2), ("dt", dt, 1), ("g", g, 2)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.device.type != "cpu" or t.dtype != torch.float32:
            raise TypeError(f"{name} must be a CPU float32 tensor, got {t.dtype} on {t.device}")
        if t.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (strides {t.stride()})")
    if int(y.shape[1]) != S:
        raise ValueError(f"y must have S={S} columns, got {int(y.shape[1])}")
    if int(g.shape[1]) != G:
        raise ValueError(f"g must have G={G} columns, got {int(g.shape[1])}")
    B = int(y.shape[0])
    if int(dt.shape[0]) != B or int(g.shape[0]) != B:
        raise ValueError(f"y, dt and g must share one batch size, got {B}, {int(dt.shape[0])}, {int(g.shape[0])}")
    if not b_min <= B <= b_max:
        raise ValueError(f"batch size {B} outside [{b_min}, {b_max}]")


def _guard_checks(case: ModelCallCase, b1_case: ModelCallCase, *, S: int, G: int, b_min: int, b_max: int) -> dict[str, Any]:
    """Run _check_step_inputs on invalid inputs (each must raise) and on a valid B=1 call (must pass)."""
    y, dt, g = case.y_phys, case.dt_seconds, case.g_phys
    B = int(y.shape[0])
    y13 = torch.cat([y, y[:, :1]], dim=1)
    big = int(b_max) + 1
    negative = {
        "float64 y": (y.double(), dt, g),
        "int64 dt": (y, dt.long(), g),
        "column-major y": (y.t().contiguous().t(), dt, g),
        "[:, :S] view of an S+1-column array": (y13[:, :S], dt, g),
        "S-1 columns": (y[:, : S - 1].contiguous(), dt, g),
        "S+1 columns": (y13.contiguous(), dt, g),
        "G+1 globals": (y, dt, torch.cat([g, g[:, :1]], dim=1).contiguous()),
        "g with B/2 rows": (y, dt, g[: max(1, B // 2)].contiguous()),
        "length-1 dt": (y, dt[:1].contiguous(), g),
        "length-1 g": (y, dt, g[:1].contiguous()),
        "0-d dt": (y, dt[0], g),
        "stride-0 dt": (y, dt[:1].expand(B), g),
        "stride-0 g": (y, dt, g[:1].expand(B, G)),
        "B = b_max + 1": (y[:1].expand(big, S).contiguous(), dt[:1].expand(big).contiguous(), g[:1].expand(big, G).contiguous()),
    }
    results: dict[str, str] = {}
    for label, (yy, dd, gg) in negative.items():
        try:
            _check_step_inputs(yy, dd, gg, S=S, G=G, b_min=b_min, b_max=b_max)
            results[label] = "ACCEPTED (guard failure)"
        except (TypeError, ValueError) as e:
            results[label] = f"refused: {e}"
    try:
        _check_step_inputs(b1_case.y_phys, b1_case.dt_seconds, b1_case.g_phys, S=S, G=G, b_min=b_min, b_max=b_max)
        b1 = "accepted"
    except (TypeError, ValueError) as e:
        b1 = f"REFUSED (guard failure): {e}"
    return {
        "negative_cases": results,
        "all_negative_refused": all(v.startswith("refused") for v in results.values()),
        "positive_b1": b1,
        "positive_b1_accepted": b1 == "accepted",
    }


def _call_model(model: Any, case: ModelCallCase) -> np.ndarray:
    """Run one direct model call on one prepared case."""

    with torch.inference_mode():
        output = model(case.y_phys, case.dt_seconds, case.g_phys)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected model output tensor, got {type(output).__name__}.")
    return output.detach().cpu().numpy()


def _compute_validation_metrics(
    *,
    comparison_name: str,
    case: ModelCallCase,
    reference_output: np.ndarray,
    candidate_output: np.ndarray,
    eps: float,
    expected_nan_rows: np.ndarray,
) -> ValidationMetrics:
    """NaN-aware comparison in float64: |log10(max(c, eps)) - log10(max(r, eps))| over the rows finite in r."""

    ref = reference_output.astype(np.float64)
    cand = candidate_output.astype(np.float64)
    nan_ref, nan_cand = np.isnan(ref), np.isnan(cand)
    rows = np.isfinite(ref).all(axis=1)
    if rows.any():
        d = np.abs(np.log10(np.maximum(cand[rows], eps)) - np.log10(np.maximum(ref[rows], eps)))
        max_dlog10 = float(np.max(d))
    else:
        max_dlog10 = float("nan")
    return ValidationMetrics(
        comparison_name=comparison_name,
        batch_size=case.batch_size,
        dt_label=case.dt_label,
        n_rows=int(ref.shape[0]),
        n_nan_rows_reference=int(nan_ref.all(axis=1).sum()),
        n_nan_rows_candidate=int(nan_cand.all(axis=1).sum()),
        nan_masks_equal=bool(np.array_equal(nan_ref, nan_cand)),
        nan_rows_match_inputs=bool(np.array_equal(nan_ref.all(axis=1), expected_nan_rows)),
        any_inf=bool(np.isinf(ref).any() or np.isinf(cand).any()),
        max_dlog10=max_dlog10,
    )


def _row_passes(metric: ValidationMetrics) -> bool:
    """The decision rule for one comparison (see the module docstring): NaN rows, inf, max dlog10."""
    in_range = metric.n_rows - metric.n_nan_rows_reference
    return (
        metric.nan_masks_equal
        and metric.nan_rows_match_inputs
        and not metric.any_inf
        and (in_range == 0 or metric.max_dlog10 <= MATCH_MAX_DLOG10)
    )


def _all_pass(metrics: list[ValidationMetrics]) -> bool:
    """Every comparison passes and together they hold at least one in-range row."""
    return all(_row_passes(m) for m in metrics) and sum(m.n_rows - m.n_nan_rows_reference for m in metrics) >= 1


def _benchmark_model(model: Any, case: ModelCallCase, *, label: str) -> BenchmarkMetrics:
    """Measure direct model-call latency for one model on one case."""

    for _ in range(BENCHMARK_WARMUP_ITERS):
        with torch.inference_mode():
            model(case.y_phys, case.dt_seconds, case.g_phys)

    wall_start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        with torch.inference_mode():
            model(case.y_phys, case.dt_seconds, case.g_phys)
    seconds_per_call = (time.perf_counter() - wall_start) / float(BENCHMARK_ITERS)
    return BenchmarkMetrics(
        model_label=label,
        batch_size=case.batch_size,
        seconds_per_call=float(seconds_per_call),
    )


def _render_markdown_report(
    *,
    setup_lines: list[str],
    interface_lines: list[str],
    decision_lines: list[str],
    validation_rows: list[ValidationMetrics],
    benchmark_rows: list[BenchmarkMetrics],
) -> str:
    """Render the markdown report written beside the AOTI package."""

    lines = ["# VULCAN AOTI Export Validation", "", "## Setup", ""]
    lines.extend(setup_lines)
    lines.extend(["", "## Interface Checks", ""])
    lines.extend(interface_lines)
    lines.extend(["", "## Decisions", ""])
    lines.extend(decision_lines)
    lines.extend(
        [
            "",
            "## Real VULCAN Input Validation",
            "",
            "| Comparison | Batch | dt | Rows | NaN rows ref | NaN rows cand | NaN masks equal | NaN rows = inputs OOR | inf | max dlog10 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in validation_rows:
        lines.append(
            f"| {row.comparison_name} | {row.batch_size} | {row.dt_label} | {row.n_rows} | {row.n_nan_rows_reference} "
            f"| {row.n_nan_rows_candidate} | {row.nan_masks_equal} | {row.nan_rows_match_inputs} | {row.any_inf} "
            f"| {row.max_dlog10:.6e} |"
        )
    lines.extend(["", "## Direct Model-Call Benchmark", "", "| Model | Batch Size | Seconds/Call |", "| --- | --- | --- |"])
    for row in benchmark_rows:
        lines.append(f"| {row.model_label} | {row.batch_size} | {row.seconds_per_call:.6e} |")
    return "\n".join(lines)


def _validate_package(
    package_path: Path,
    *,
    export_impl: Any,
    fresh_export: ExportBuildContext,
    installed_raw_model: Any,
    installed_metadata: dict[str, Any],
    cases: list[ModelCallCase],
    benchmark_cases: list[ModelCallCase],
    b1_case: ModelCallCase,
    guard_case: ModelCallCase,
) -> dict[str, Any]:
    """Load the staged package once (default thread-safe mode), compare, check the guard, benchmark."""
    meta = fresh_export.metadata
    S, G = len(meta["species_variables"]), len(meta["global_variables"])
    b_min, b_max = int(meta["dynamic_batch"]["min"]), int(meta["dynamic_batch"]["max"])
    eps = float(meta["epsilon"])
    compiled_model = _inductor.aoti_load_package(str(package_path))

    aoti_rows: list[ValidationMetrics] = []
    installed_rows: list[ValidationMetrics] = []
    for case in cases:
        # The guard runs once per case, outside the timed loops.
        _check_step_inputs(case.y_phys, case.dt_seconds, case.g_phys, S=S, G=G, b_min=b_min, b_max=b_max)
        expected = export_impl.input_out_of_range_rows(
            case.y_phys.numpy(), case.dt_seconds.numpy(), case.g_phys.numpy(), meta
        )
        fresh_output = _call_model(fresh_export.fresh_raw_model, case)
        aoti_rows.append(_compute_validation_metrics(
            comparison_name="aoti_vs_fresh_raw", case=case, reference_output=fresh_output,
            candidate_output=_call_model(compiled_model, case), eps=eps, expected_nan_rows=expected))
        if case.batch_size == b_max:  # the B_MAX case checks the package; the installed model may predate it
            continue
        installed_rows.append(_compute_validation_metrics(
            comparison_name="installed_raw_vs_fresh_raw", case=case, reference_output=fresh_output,
            candidate_output=_call_model(installed_raw_model, case), eps=eps, expected_nan_rows=expected))

    guard = _guard_checks(guard_case, b1_case, S=S, G=G, b_min=b_min, b_max=b_max)

    benchmark_rows: list[BenchmarkMetrics] = []
    for case in benchmark_cases:
        benchmark_rows.append(_benchmark_model(fresh_export.fresh_raw_model, case, label="fresh_raw_export"))
        benchmark_rows.append(_benchmark_model(installed_raw_model, case, label="installed_raw_export"))
        benchmark_rows.append(_benchmark_model(compiled_model, case, label="aoti_package"))

    failing = [r for r in aoti_rows if not _row_passes(r)]
    passed = _all_pass(aoti_rows) and guard["all_negative_refused"] and guard["positive_b1_accepted"]
    return {
        "aoti_rows": aoti_rows,
        "installed_rows": installed_rows,
        "guard": guard,
        "benchmark_rows": benchmark_rows,
        "failing": failing,
        "passed": passed,
        # A pre-NaN-row installed model clamps out-of-range rows instead of NaN-marking them.
        "installed_has_nan_row_contract": installed_metadata.get("oor_policy") == "nan_row",
    }


def main() -> None:
    """Build the VULCAN-specific AOTI package, validate it, benchmark it, and install it only on a pass."""

    export_impl = _load_export_impl()
    run_dir = export_impl.resolve_run_dir()
    out_package = run_dir / OUTPUT_AOTI_PACKAGE_NAME
    out_report = run_dir / OUTPUT_REPORT_NAME
    out_json = run_dir / OUTPUT_JSON_NAME

    reference_column = _load_reference_column(REFERENCE_VULCAN_STATE_PATH)
    # Validate both required external inputs before the expensive export/compile.
    if not INSTALLED_RAW_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Installed raw VULCAN model is required for this script and is missing: "
            f"{INSTALLED_RAW_MODEL_PATH}\n"
            "This is an external, machine-specific input from a VULCAN working tree; "
            "point INSTALLED_RAW_MODEL_PATH at the installed Emulator/model/model.pt2."
        )
    example_batch = _resolve_example_batch(reference_column, min_example_batch=export_impl.MIN_EXAMPLE_BATCH)
    batch_sizes = _resolve_benchmark_batch_sizes(reference_column)

    fresh_export = _build_fresh_raw_export(export_impl, run_dir=run_dir, example_batch=example_batch)
    meta = fresh_export.metadata

    installed_raw_model, installed_metadata = _load_raw_export(INSTALLED_RAW_MODEL_PATH)
    installed_model_species = tuple(_strip_export_suffix(str(name)) for name in installed_metadata.get("species_variables", []))
    installed_global_variables = tuple(str(name) for name in installed_metadata.get("global_variables", []))
    if installed_model_species != fresh_export.model_species:
        raise ValueError(
            "Installed raw model species do not match the fresh raw export species order: "
            f"installed={installed_model_species}, fresh={fresh_export.model_species}"
        )
    if installed_global_variables != fresh_export.global_variables:
        raise ValueError(
            "Installed raw model globals do not match the fresh raw export globals: "
            f"installed={installed_global_variables}, fresh={fresh_export.global_variables}"
        )

    y_rows, g_rows, composition = _column_inputs(
        reference_column, model_species=fresh_export.model_species, floor=float(meta["epsilon"]))
    # The largest exported batch is validated too (not benchmarked).
    b_max = int(meta["dynamic_batch"]["max"])
    validation_batch_sizes = batch_sizes + (() if b_max in batch_sizes else (b_max,))
    dts = {
        "dt_min": float(meta["dt_min_seconds"]),
        "host_dt": HOST_DT_SECONDS,
        "dt_max": float(meta["dt_max_seconds"]),
    }
    cases = _build_model_call_cases(
        reference_column, y_rows=y_rows, g_rows=g_rows, batch_sizes=validation_batch_sizes, dts=dts)
    guard_case = next(c for c in cases if c.batch_size == reference_column.nz and c.dt_label == "host_dt")
    oor_dt = guard_case.dt_seconds.clone()
    oor_dt[0] = float(meta["dt_max_seconds"]) * OOR_PROBE_DT_FACTOR
    cases.append(ModelCallCase(batch_size=guard_case.batch_size, dt_label="host_dt + one dt above dt_max",
                               y_phys=guard_case.y_phys, dt_seconds=oor_dt, g_phys=guard_case.g_phys))
    benchmark_cases = [c for c in cases if c.dt_label == "host_dt" and c.batch_size in batch_sizes]
    b1_case = next(c for c in cases if c.batch_size == 1) if 1 in batch_sizes else ModelCallCase(
        batch_size=1, dt_label="host_dt", y_phys=guard_case.y_phys[:1].contiguous(),
        dt_seconds=guard_case.dt_seconds[:1].contiguous(), g_phys=guard_case.g_phys[:1].contiguous())

    outcome: dict[str, Any] = {}

    def verify(staging: Path) -> None:
        outcome.update(_validate_package(
            staging, export_impl=export_impl, fresh_export=fresh_export, installed_raw_model=installed_raw_model,
            installed_metadata=installed_metadata, cases=cases, benchmark_cases=benchmark_cases,
            b1_case=b1_case, guard_case=guard_case))
        if not outcome["passed"]:
            raise _ValidationFailed()

    installed_package = True
    try:
        export_impl.save_artifact(
            out_package,
            lambda staging: _compile_aoti_package(fresh_export.exported_program, metadata=meta, path=staging),
            verify=verify,
        )
    except _ValidationFailed:
        installed_package = False

    aoti_rows: list[ValidationMetrics] = outcome["aoti_rows"]
    installed_rows: list[ValidationMetrics] = outcome["installed_rows"]
    benchmark_rows: list[BenchmarkMetrics] = outcome["benchmark_rows"]
    at_nz = {r.model_label: r for r in benchmark_rows if r.batch_size == reference_column.nz}
    installed_matches = (
        _all_pass(installed_rows) if outcome["installed_has_nan_row_contract"]
        else "not comparable: the installed model predates the NaN-row contract (it clamps out-of-range rows)"
    )
    worst = max(aoti_rows, key=lambda r: (not _row_passes(r), r.max_dlog10 if r.max_dlog10 == r.max_dlog10 else math.inf))

    report_payload = {
        "setup": {
            "run_dir": str(run_dir),
            "checkpoint": export_impl.CHECKPOINT,
            "reference_vulcan_state_path": str(REFERENCE_VULCAN_STATE_PATH),
            "installed_raw_model_path": str(INSTALLED_RAW_MODEL_PATH),
            "output_aoti_package_path": str(out_package),
            "package_installed": installed_package,
            "output_report_path": str(out_report),
            "output_json_path": str(out_json),
            "reference_nz": reference_column.nz,
            "aoti_example_batch": example_batch,
            "benchmark_batch_sizes": list(batch_sizes),
            "validation_batch_sizes": list(validation_batch_sizes),
            "validation_dt_seconds": dts,
            "oor_probe_dt_seconds": float(oor_dt[0]),
            "match_max_dlog10": MATCH_MAX_DLOG10,
            "build_target": meta["build_target"],
            "build_host": meta["build_host"],
        },
        "interface_checks": {
            "model_species": list(fresh_export.model_species),
            "global_variables": list(fresh_export.global_variables),
            "fresh_raw_export_dtype": meta["export_dtype"],
            "fresh_raw_dt_min_seconds": meta["dt_min_seconds"],
            "fresh_raw_dt_max_seconds": meta["dt_max_seconds"],
            "fresh_raw_epsilon": meta["epsilon"],
            "installed_raw_export_dtype": installed_metadata.get("export_dtype"),
            "installed_raw_dt_min_seconds": installed_metadata.get("dt_min_seconds"),
            "installed_raw_dt_max_seconds": installed_metadata.get("dt_max_seconds"),
            "installed_raw_epsilon": installed_metadata.get("epsilon"),
            "installed_raw_oor_policy": installed_metadata.get("oor_policy"),
            "input_guard": outcome["guard"],
            "column_composition": composition,
            "training_species_sum": meta.get("species_sum"),
        },
        "decisions": {
            "match_decision_rule": (
                "per case: identical NaN entries, NaN rows equal to the inputs' out-of-range rows, no +-inf, "
                f"max |dlog10| <= {MATCH_MAX_DLOG10:.1e} over the in-range rows (eps = metadata epsilon); >= 1 in-range "
                "row over all cases; plus every invalid input refused by the guard and B=1 accepted"
            ),
            "aoti_matches_fresh_raw_on_vulcan_inputs": outcome["passed"],
            "worst_aoti_case": asdict(worst),
            "installed_raw_matches_fresh_raw_on_vulcan_inputs (informational)": installed_matches,
            "aoti_minus_installed_raw_seconds_at_nz": (
                at_nz["aoti_package"].seconds_per_call - at_nz["installed_raw_export"].seconds_per_call
            ),
        },
        "validation_rows": [asdict(row) for row in aoti_rows + installed_rows],
        "benchmark_rows": [asdict(row) for row in benchmark_rows],
    }
    # JSON first: a formatting error in the markdown below cannot lose the results.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    tsum = composition["tracked_sum"]
    h2 = composition["h2_share_of_sum"]
    setup_lines = [f"- {k}: `{v}`" for k, v in report_payload["setup"].items()]
    interface_lines = [
        f"- model_species: {', '.join(fresh_export.model_species)}",
        f"- global_variables: {', '.join(fresh_export.global_variables)}",
        f"- fresh_raw_export_dtype: {meta['export_dtype']}; dt range [{meta['dt_min_seconds']:.6e}, {meta['dt_max_seconds']:.6e}] s; "
        f"epsilon {meta['epsilon']:g}",
        f"- installed_raw: dtype {installed_metadata.get('export_dtype')}, oor_policy {installed_metadata.get('oor_policy')}, "
        f"epsilon {installed_metadata.get('epsilon')}",
        f"- input guard: all invalid inputs refused = {outcome['guard']['all_negative_refused']}; B=1 {outcome['guard']['positive_b1']}",
        f"- column tracked-species sum min/median/max: {tsum['min']:.4g} / {tsum['median']:.4g} / {tsum['max']:.4g} "
        f"(training: {meta.get('species_sum', {}).get('quantiles') if isinstance(meta.get('species_sum'), dict) else meta.get('species_sum')})",
        "- column H2 share of the sum min/median/max: "
        + ("n/a" if h2 is None else f"{h2['min']:.4g} / {h2['median']:.4g} / {h2['max']:.4g}"),
    ]
    decision_lines = [f"- {k}: {v}" for k, v in report_payload["decisions"].items()]
    markdown_report = _render_markdown_report(
        setup_lines=setup_lines,
        interface_lines=interface_lines,
        decision_lines=decision_lines,
        validation_rows=aoti_rows + installed_rows,
        benchmark_rows=benchmark_rows,
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(markdown_report, encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_report}")
    print(f"aoti_matches_fresh_raw_on_vulcan_inputs: {outcome['passed']}")
    print(f"installed_raw_matches_fresh_raw_on_vulcan_inputs (informational): {installed_matches}")
    if not installed_package:
        bad = outcome["failing"][0] if outcome["failing"] else worst
        raise SystemExit(
            f"AOTI validation failed (package not installed; {out_package} unchanged): "
            f"{bad.comparison_name} batch={bad.batch_size} dt={bad.dt_label} max_dlog10={bad.max_dlog10:.3e} "
            f"nan_masks_equal={bad.nan_masks_equal} nan_rows_match_inputs={bad.nan_rows_match_inputs}; "
            f"input guard ok={outcome['guard']['all_negative_refused'] and outcome['guard']['positive_b1_accepted']}"
        )
    print(f"wrote {out_package}")


if __name__ == "__main__":
    main()
