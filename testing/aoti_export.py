#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and validate a VULCAN-oriented CPU AOTInductor package.

This script is validation-first:
- it builds one CPU AOTInductor package from the current ``v3`` checkpoint
- it validates that package against both a fresh raw export and the installed
  raw VULCAN model on real VULCAN column inputs
- it benchmarks the direct model-call latency for the raw and AOTI paths
- it writes a markdown report plus a JSON report

It does not overwrite ``Emulator/model/model.pt2``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import _inductor
from torch._export.serde.serialize import SerializedArtifact, deserialize
from torch.export import Dim


ROOT = Path(__file__).resolve().parents[2]
EXPORT_IMPL_PATH = (ROOT / "Auto-Chem" / "testing" / "export.py").resolve()
RUN_DIR = (ROOT / "Auto-Chem" / "models" / "v3").resolve()
CHECKPOINT = "checkpoints/last.ckpt"
REFERENCE_VULCAN_STATE_PATH = (ROOT / "output" / "HD189_short_thermo.vul").resolve()
INSTALLED_RAW_MODEL_PATH = (ROOT / "Emulator" / "model" / "model.pt2").resolve()
OUTPUT_AOTI_PACKAGE_PATH = (
    ROOT / "Auto-Chem" / "models" / "v3" / "export_cpu_dynB_1step_phys_aoti.pt2"
).resolve()
OUTPUT_REPORT_PATH = (
    ROOT / "Auto-Chem" / "models" / "v3" / "export_cpu_dynB_1step_phys_aoti_validation.md"
).resolve()
OUTPUT_JSON_PATH = (
    ROOT / "Auto-Chem" / "models" / "v3" / "export_cpu_dynB_1step_phys_aoti_validation.json"
).resolve()
AOTI_EXAMPLE_BATCH: int | None = None
BENCHMARK_BATCH_SIZES: tuple[int | str, ...] = (1, "nz", "2*nz")
TEST_RUN_SINGLE_THREADED = (False, True)
VALIDATION_DT_SECONDS = 10.0
INPUT_FLOOR = 1.0e-15
EPSILON = 1.0e-30
BENCHMARK_WARMUP_ITERS = 3
BENCHMARK_ITERS = 20
MATCH_MAX_ABS_TOLERANCE = 1.0e-6
MATCH_P90_REL_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class ExportBuildContext:
    """Fresh raw export objects created from the current export code path."""

    exported_program: torch.export.ExportedProgram
    fresh_raw_model: torch.nn.Module
    metadata: dict[str, Any]
    model_species: tuple[str, ...]
    global_variables: tuple[str, ...]
    dtype: torch.dtype


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
    """One prepared model-call case at a specific batch size."""

    batch_size: int
    y_phys: torch.Tensor  # shape: (B, S)
    dt_seconds: torch.Tensor  # shape: (B,)
    g_phys: torch.Tensor  # shape: (B, 2)


@dataclass(frozen=True)
class ValidationMetrics:
    """Difference metrics for one reference-vs-candidate model comparison."""

    comparison_name: str
    batch_size: int
    max_abs_diff: float
    median_rel_diff: float
    p90_rel_diff: float
    max_rel_diff: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Direct model-call wall-clock timing for one model and batch size."""

    model_label: str
    batch_size: int
    seconds_per_call: float


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


def _read_embedded_metadata_json(path: Path) -> dict[str, Any]:
    """Read embedded metadata from either raw-export or AOTI package layouts."""

    if not path.exists():
        raise FileNotFoundError(f"Missing export artifact: {path}")

    with zipfile.ZipFile(path, "r") as archive:
        archive_names = archive.namelist()
        if not archive_names:
            raise RuntimeError(f"Export archive is empty: {path}")

        candidate_names = ["extra/metadata.json"]
        candidate_names.extend(
            name for name in archive_names if name.endswith("/extra/metadata.json")
        )
        for candidate_name in candidate_names:
            try:
                raw_metadata = archive.read(candidate_name).decode("utf-8")
            except KeyError:
                continue
            metadata = json.loads(raw_metadata)
            if not isinstance(metadata, dict):
                raise TypeError(f"Embedded metadata must be a JSON object in {path}.")
            return metadata

    raise FileNotFoundError(f"Embedded metadata.json is missing from {path}.")


def _load_raw_model_from_pt2(path: Path, *, dtype: torch.dtype) -> torch.nn.Module:
    """Load one raw ``torch.export.save(...)`` artifact as a callable module."""

    if not path.exists():
        raise FileNotFoundError(f"Missing raw export artifact: {path}")

    with zipfile.ZipFile(path, "r") as archive:
        archive_names = archive.namelist()
        if not archive_names:
            raise RuntimeError(f"Export archive is empty: {path}")
        archive_root = archive_names[0].split("/", 1)[0] + "/"
        artifact = SerializedArtifact(
            exported_program=archive.read(f"{archive_root}models/model.json"),
            state_dict=archive.read(f"{archive_root}data/weights/model.pt"),
            constants=archive.read(f"{archive_root}data/constants/model.pt"),
            example_inputs=archive.read(f"{archive_root}data/sample_inputs/model.pt"),
        )

    return deserialize(artifact).module().to(dtype=dtype)


def _load_reference_column(path: Path) -> ReferenceColumn:
    """Load the default VULCAN column state used for real-input validation."""

    if not path.exists():
        raise FileNotFoundError(
            "Reference VULCAN state is required for this script and is missing: "
            f"{path}"
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


def _resolve_example_batch(reference_column: ReferenceColumn) -> int:
    """Resolve the AOTI export example batch size."""

    if AOTI_EXAMPLE_BATCH is None:
        return reference_column.nz
    return int(AOTI_EXAMPLE_BATCH)


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


def _build_fresh_raw_export(
    export_impl: Any,
    *,
    example_batch: int,
) -> ExportBuildContext:
    """Build the fresh raw export in memory from the current export code path."""

    dtype = export_impl._parse_dtype(export_impl.EXPORT_DTYPE)
    cfg, cfg_path = export_impl._load_resolved_config(RUN_DIR)
    ckpt_path = export_impl._resolve_checkpoint_path(CHECKPOINT, cfg_path=cfg_path)
    processed_dir = export_impl._resolve_processed_dir(cfg, cfg_path=cfg_path)
    manifest_path = processed_dir / "normalization.json"
    manifest = export_impl._load_json(manifest_path)
    species_vars, global_vars = export_impl._validate_manifest_vs_config(cfg, manifest)

    base_cpu = export_impl.create_model(cfg)
    export_impl._load_weights_strict(base_cpu, ckpt_path)
    base_cpu = export_impl._freeze_for_inference(
        base_cpu.to(device=torch.device("cpu"), dtype=dtype)
    )

    norm_cpu = export_impl.build_baked_normalizer(
        manifest,
        species_vars=species_vars,
        global_vars=global_vars,
    )
    norm_cpu = norm_cpu.to(device=torch.device("cpu"), dtype=dtype)
    norm_cpu.eval()

    step = export_impl.OneStepPhysical(base_cpu, norm_cpu)
    step = export_impl._freeze_for_inference(step)

    example_inputs = export_impl._make_example_inputs(
        norm_cpu,
        B=max(1, int(example_batch)),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    batch_dim = Dim("B", min=int(export_impl.B_MIN), max=int(export_impl.B_MAX))
    dynamic_shapes = ({0: batch_dim}, {0: batch_dim}, {0: batch_dim})
    exported_program = torch.export.export(
        step,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=bool(export_impl.EXPORT_STRICT),
    )
    export_impl._verify_dynamic_batch(
        exported_program,
        device=torch.device("cpu"),
        dtype=dtype,
        norm=norm_cpu,
    )

    metadata = {
        "format": "1step_physical_dynB",
        "run_dir": export_impl._to_repo_relative_str(RUN_DIR),
        "config_path": export_impl._to_repo_relative_str(cfg_path),
        "checkpoint_path": export_impl._to_repo_relative_str(ckpt_path),
        "normalization_path": export_impl._to_repo_relative_str(manifest_path),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_visible_devices": "",
        "export_device": "cpu",
        "export_device_tag": "cpu",
        "export_dtype": str(dtype).replace("torch.", ""),
        "species_variables": list(species_vars),
        "global_variables": list(global_vars),
        "normalization_methods": dict(manifest["normalization_methods"]),
        "epsilon": float(manifest.get("epsilon", 1e-30)),
        "dt_log10_min": float(manifest["dt"]["log_min"]),
        "dt_log10_max": float(manifest["dt"]["log_max"]),
        "dt_min_seconds": float(10.0 ** float(manifest["dt"]["log_min"])),
        "dt_max_seconds": float(10.0 ** float(manifest["dt"]["log_max"])),
        "dynamic_batch": {"min": int(export_impl.B_MIN), "max": int(export_impl.B_MAX)},
        "signature": {
            "inputs": {"y_phys": ["B", "S"], "dt_seconds": ["B"], "g_phys": ["B", "G"]},
            "output": {"y_next_phys": ["B", "S"]},
        },
        "aoti_example_batch": int(example_batch),
    }

    return ExportBuildContext(
        exported_program=exported_program,
        fresh_raw_model=exported_program.module(),
        metadata=metadata,
        model_species=tuple(_strip_export_suffix(name) for name in species_vars),
        global_variables=tuple(str(name) for name in global_vars),
        dtype=dtype,
    )


def _write_aoti_package(
    exported_program: torch.export.ExportedProgram,
    *,
    metadata: dict[str, Any],
    out_path: Path,
) -> Path:
    """Compile and save the CPU AOTI package plus embedded metadata."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    compiled_path = Path(
        _inductor.aoti_compile_and_package(exported_program, package_path=str(out_path))
    ).resolve()
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
    return compiled_path


def _build_model_call_cases(
    reference_column: ReferenceColumn,
    *,
    model_species: tuple[str, ...],
    batch_sizes: tuple[int, ...],
) -> list[ModelCallCase]:
    """Build real VULCAN model-call cases in reduced-species physical space."""

    species_index = {
        species_name: position for position, species_name in enumerate(reference_column.species)
    }
    active_indices = np.asarray(
        [species_index[species_name] for species_name in model_species],
        dtype=int,
    )

    total_density = np.sum(reference_column.y_full, axis=1, keepdims=True)
    y_active = reference_column.y_full[:, active_indices]
    ymix_active = y_active / np.maximum(total_density, EPSILON)
    ymix_sanitized = np.maximum(
        np.clip(ymix_active.astype(np.float32), 0.0, None),
        INPUT_FLOOR,
    )  # shape: (nz, n_model_species)
    pco_barye = reference_column.pco_barye.astype(np.float32)
    Tco_K = reference_column.Tco_K.astype(np.float32)
    g_full = np.column_stack((pco_barye, Tco_K)).astype(np.float32)  # shape: (nz, 2)

    cases: list[ModelCallCase] = []
    for batch_size in batch_sizes:
        repeat_count = int(math.ceil(float(batch_size) / float(reference_column.nz)))
        y_case = np.concatenate([ymix_sanitized] * repeat_count, axis=0)[:batch_size].copy()
        g_case = np.concatenate([g_full] * repeat_count, axis=0)[:batch_size].copy()
        dt_case = np.full((batch_size,), VALIDATION_DT_SECONDS, dtype=np.float32)
        cases.append(
            ModelCallCase(
                batch_size=int(batch_size),
                y_phys=torch.from_numpy(y_case),
                dt_seconds=torch.from_numpy(dt_case),
                g_phys=torch.from_numpy(g_case),
            )
        )
    return cases


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
    batch_size: int,
    reference_output: np.ndarray,
    candidate_output: np.ndarray,
) -> ValidationMetrics:
    """Compute absolute and relative difference metrics for one comparison."""

    abs_diff = np.abs(candidate_output - reference_output)
    rel_diff = abs_diff / np.maximum(np.abs(reference_output), EPSILON)
    return ValidationMetrics(
        comparison_name=comparison_name,
        batch_size=int(batch_size),
        max_abs_diff=float(np.max(abs_diff)),
        median_rel_diff=float(np.median(rel_diff)),
        p90_rel_diff=float(np.percentile(rel_diff, 90)),
        max_rel_diff=float(np.max(rel_diff)),
    )


def _matches_reference(metrics: list[ValidationMetrics]) -> bool:
    """Apply the report decision rule for output matching."""

    return all(
        metric.max_abs_diff <= MATCH_MAX_ABS_TOLERANCE
        and metric.p90_rel_diff <= MATCH_P90_REL_TOLERANCE
        for metric in metrics
    )


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

    lines = [
        "# VULCAN AOTI Export Validation",
        "",
        "## Setup",
        "",
    ]
    lines.extend(setup_lines)
    lines.extend(
        [
            "",
            "## Interface Checks",
            "",
        ]
    )
    lines.extend(interface_lines)
    lines.extend(
        [
            "",
            "## Decisions",
            "",
        ]
    )
    lines.extend(decision_lines)
    lines.extend(
        [
            "",
            "## Real VULCAN Input Validation",
            "",
            "| Comparison | Batch Size | Max AbsDiff | Median RelDiff | P90 RelDiff | Max RelDiff |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in validation_rows:
        lines.append(
            "| "
            + row.comparison_name
            + " | "
            + str(row.batch_size)
            + " | "
            + f"{row.max_abs_diff:.6e}"
            + " | "
            + f"{row.median_rel_diff:.6e}"
            + " | "
            + f"{row.p90_rel_diff:.6e}"
            + " | "
            + f"{row.max_rel_diff:.6e}"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Direct Model-Call Benchmark",
            "",
            "| Model | Batch Size | Seconds/Call |",
            "| --- | --- | --- |",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| "
            + row.model_label
            + " | "
            + str(row.batch_size)
            + " | "
            + f"{row.seconds_per_call:.6e}"
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Build the VULCAN-specific AOTI package, validate it, and benchmark it."""

    export_impl = _load_export_impl()
    reference_column = _load_reference_column(REFERENCE_VULCAN_STATE_PATH)
    resolved_example_batch = _resolve_example_batch(reference_column)
    resolved_batch_sizes = _resolve_benchmark_batch_sizes(reference_column)

    fresh_export = _build_fresh_raw_export(
        export_impl,
        example_batch=resolved_example_batch,
    )
    compiled_package_path = _write_aoti_package(
        fresh_export.exported_program,
        metadata=fresh_export.metadata,
        out_path=OUTPUT_AOTI_PACKAGE_PATH,
    )

    installed_metadata = _read_embedded_metadata_json(INSTALLED_RAW_MODEL_PATH)
    installed_model_species = tuple(
        _strip_export_suffix(str(name))
        for name in installed_metadata.get("species_variables", [])
    )
    installed_global_variables = tuple(
        str(name) for name in installed_metadata.get("global_variables", [])
    )
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

    installed_raw_model = _load_raw_model_from_pt2(
        INSTALLED_RAW_MODEL_PATH,
        dtype=fresh_export.dtype,
    )
    compiled_models = {
        run_single_threaded: _inductor.aoti_load_package(
            str(compiled_package_path),
            run_single_threaded=run_single_threaded,
        )
        for run_single_threaded in TEST_RUN_SINGLE_THREADED
    }

    cases = _build_model_call_cases(
        reference_column,
        model_species=fresh_export.model_species,
        batch_sizes=resolved_batch_sizes,
    )

    validation_rows: list[ValidationMetrics] = []
    benchmark_rows: list[BenchmarkMetrics] = []
    fresh_outputs_by_batch: dict[int, np.ndarray] = {}
    installed_outputs_by_batch: dict[int, np.ndarray] = {}
    compiled_outputs_by_mode_and_batch: dict[tuple[bool, int], np.ndarray] = {}

    for case in cases:
        fresh_output = _call_model(fresh_export.fresh_raw_model, case)
        installed_output = _call_model(installed_raw_model, case)
        fresh_outputs_by_batch[case.batch_size] = fresh_output
        installed_outputs_by_batch[case.batch_size] = installed_output
        validation_rows.append(
            _compute_validation_metrics(
                comparison_name="installed_raw_vs_fresh_raw",
                batch_size=case.batch_size,
                reference_output=fresh_output,
                candidate_output=installed_output,
            )
        )
        for run_single_threaded, compiled_model in compiled_models.items():
            compiled_output = _call_model(compiled_model, case)
            compiled_outputs_by_mode_and_batch[(run_single_threaded, case.batch_size)] = compiled_output
            validation_rows.append(
                _compute_validation_metrics(
                    comparison_name=(
                        "aoti_run_single_threaded_true_vs_fresh_raw"
                        if run_single_threaded
                        else "aoti_run_single_threaded_false_vs_fresh_raw"
                    ),
                    batch_size=case.batch_size,
                    reference_output=fresh_output,
                    candidate_output=compiled_output,
                )
            )

    for case in cases:
        benchmark_rows.append(
            _benchmark_model(
                fresh_export.fresh_raw_model,
                case,
                label="fresh_raw_export",
            )
        )
        benchmark_rows.append(
            _benchmark_model(
                installed_raw_model,
                case,
                label="installed_raw_export",
            )
        )
        for run_single_threaded, compiled_model in compiled_models.items():
            benchmark_rows.append(
                _benchmark_model(
                    compiled_model,
                    case,
                    label=(
                        "aoti_package_run_single_threaded_true"
                        if run_single_threaded
                        else "aoti_package_run_single_threaded_false"
                    ),
                )
            )

    installed_vs_fresh_rows = [
        row for row in validation_rows if row.comparison_name == "installed_raw_vs_fresh_raw"
    ]
    aoti_vs_fresh_rows = [
        row for row in validation_rows if row.comparison_name != "installed_raw_vs_fresh_raw"
    ]
    installed_raw_matches_fresh_raw = _matches_reference(installed_vs_fresh_rows)
    aoti_matches_fresh_raw = _matches_reference(aoti_vs_fresh_rows)

    benchmark_rows_at_nz = [
        row for row in benchmark_rows if row.batch_size == reference_column.nz
    ]
    if not benchmark_rows_at_nz:
        raise RuntimeError("Benchmark results for batch size nz are missing.")

    installed_raw_at_nz = next(
        row for row in benchmark_rows_at_nz if row.model_label == "installed_raw_export"
    )
    aoti_rows_at_nz = [
        row for row in benchmark_rows_at_nz if row.model_label.startswith("aoti_package_")
    ]
    fastest_aoti_at_nz = min(aoti_rows_at_nz, key=lambda row: row.seconds_per_call)
    fastest_overall_at_nz = min(benchmark_rows_at_nz, key=lambda row: row.seconds_per_call)
    aoti_faster_than_installed_raw_at_nz = (
        fastest_aoti_at_nz.seconds_per_call < installed_raw_at_nz.seconds_per_call
    )

    installed_dtype_matches_fresh = (
        str(installed_metadata.get("export_dtype", "")).strip().lower()
        == str(fresh_export.metadata["export_dtype"]).strip().lower()
    )
    installed_dt_min_matches_fresh = (
        float(installed_metadata.get("dt_min_seconds", float("nan")))
        == float(fresh_export.metadata["dt_min_seconds"])
    )
    installed_dt_max_matches_fresh = (
        float(installed_metadata.get("dt_max_seconds", float("nan")))
        == float(fresh_export.metadata["dt_max_seconds"])
    )

    report_payload = {
        "setup": {
            "run_dir": str(RUN_DIR),
            "checkpoint": CHECKPOINT,
            "reference_vulcan_state_path": str(REFERENCE_VULCAN_STATE_PATH),
            "installed_raw_model_path": str(INSTALLED_RAW_MODEL_PATH),
            "output_aoti_package_path": str(compiled_package_path),
            "output_report_path": str(OUTPUT_REPORT_PATH),
            "output_json_path": str(OUTPUT_JSON_PATH),
            "reference_nz": reference_column.nz,
            "resolved_aoti_example_batch": resolved_example_batch,
            "benchmark_batch_sizes": list(resolved_batch_sizes),
            "test_run_single_threaded": list(TEST_RUN_SINGLE_THREADED),
            "validation_dt_seconds": VALIDATION_DT_SECONDS,
            "match_max_abs_tolerance": MATCH_MAX_ABS_TOLERANCE,
            "match_p90_rel_tolerance": MATCH_P90_REL_TOLERANCE,
        },
        "interface_checks": {
            "model_species": list(fresh_export.model_species),
            "global_variables": list(fresh_export.global_variables),
            "fresh_raw_export_dtype": fresh_export.metadata["export_dtype"],
            "fresh_raw_dt_min_seconds": fresh_export.metadata["dt_min_seconds"],
            "fresh_raw_dt_max_seconds": fresh_export.metadata["dt_max_seconds"],
            "installed_raw_export_dtype": installed_metadata.get("export_dtype"),
            "installed_raw_dt_min_seconds": installed_metadata.get("dt_min_seconds"),
            "installed_raw_dt_max_seconds": installed_metadata.get("dt_max_seconds"),
            "installed_raw_species_match_fresh": installed_model_species == fresh_export.model_species,
            "installed_raw_globals_match_fresh": installed_global_variables == fresh_export.global_variables,
            "installed_raw_dtype_match_fresh": installed_dtype_matches_fresh,
            "installed_raw_dt_min_match_fresh": installed_dt_min_matches_fresh,
            "installed_raw_dt_max_match_fresh": installed_dt_max_matches_fresh,
        },
        "decisions": {
            "match_decision_rule": (
                "match means max_abs_diff <= "
                f"{MATCH_MAX_ABS_TOLERANCE:.1e} and p90_rel_diff <= "
                f"{MATCH_P90_REL_TOLERANCE:.1e} for every real-input comparison row"
            ),
            "installed_raw_matches_fresh_raw_on_vulcan_inputs": installed_raw_matches_fresh_raw,
            "aoti_matches_fresh_raw_on_vulcan_inputs": aoti_matches_fresh_raw,
            "fastest_aoti_mode_at_nz": fastest_aoti_at_nz.model_label,
            "fastest_model_at_nz": fastest_overall_at_nz.model_label,
            "aoti_faster_than_installed_raw_at_nz": aoti_faster_than_installed_raw_at_nz,
            "aoti_minus_installed_raw_seconds_at_nz": (
                fastest_aoti_at_nz.seconds_per_call - installed_raw_at_nz.seconds_per_call
            ),
        },
        "validation_rows": [asdict(row) for row in validation_rows],
        "benchmark_rows": [asdict(row) for row in benchmark_rows],
    }

    setup_lines = [
        f"- run_dir: `{RUN_DIR}`",
        f"- checkpoint: `{CHECKPOINT}`",
        f"- reference_vulcan_state_path: `{REFERENCE_VULCAN_STATE_PATH}`",
        f"- installed_raw_model_path: `{INSTALLED_RAW_MODEL_PATH}`",
        f"- output_aoti_package_path: `{compiled_package_path}`",
        f"- output_report_path: `{OUTPUT_REPORT_PATH}`",
        f"- output_json_path: `{OUTPUT_JSON_PATH}`",
        f"- reference_nz: {reference_column.nz}",
        f"- resolved_aoti_example_batch: {resolved_example_batch}",
        f"- benchmark_batch_sizes: {', '.join(str(value) for value in resolved_batch_sizes)}",
        f"- test_run_single_threaded: {', '.join(str(value) for value in TEST_RUN_SINGLE_THREADED)}",
        f"- validation_dt_seconds: {VALIDATION_DT_SECONDS:.6e}",
        f"- match_max_abs_tolerance: {MATCH_MAX_ABS_TOLERANCE:.6e}",
        f"- match_p90_rel_tolerance: {MATCH_P90_REL_TOLERANCE:.6e}",
    ]
    interface_lines = [
        f"- model_species: {', '.join(fresh_export.model_species)}",
        f"- global_variables: {', '.join(fresh_export.global_variables)}",
        f"- fresh_raw_export_dtype: {fresh_export.metadata['export_dtype']}",
        f"- fresh_raw_dt_min_seconds: {float(fresh_export.metadata['dt_min_seconds']):.6e}",
        f"- fresh_raw_dt_max_seconds: {float(fresh_export.metadata['dt_max_seconds']):.6e}",
        f"- installed_raw_export_dtype: {installed_metadata.get('export_dtype')}",
        f"- installed_raw_dt_min_seconds: {float(installed_metadata.get('dt_min_seconds')):.6e}",
        f"- installed_raw_dt_max_seconds: {float(installed_metadata.get('dt_max_seconds')):.6e}",
        f"- installed_raw_species_match_fresh: {installed_model_species == fresh_export.model_species}",
        f"- installed_raw_globals_match_fresh: {installed_global_variables == fresh_export.global_variables}",
        f"- installed_raw_dtype_match_fresh: {installed_dtype_matches_fresh}",
        f"- installed_raw_dt_min_match_fresh: {installed_dt_min_matches_fresh}",
        f"- installed_raw_dt_max_match_fresh: {installed_dt_max_matches_fresh}",
    ]
    decision_lines = [
        f"- match_decision_rule: {report_payload['decisions']['match_decision_rule']}",
        f"- installed_raw_matches_fresh_raw_on_vulcan_inputs: {installed_raw_matches_fresh_raw}",
        f"- aoti_matches_fresh_raw_on_vulcan_inputs: {aoti_matches_fresh_raw}",
        f"- fastest_aoti_mode_at_nz: {fastest_aoti_at_nz.model_label}",
        f"- fastest_model_at_nz: {fastest_overall_at_nz.model_label}",
        f"- aoti_faster_than_installed_raw_at_nz: {aoti_faster_than_installed_raw_at_nz}",
        (
            "- aoti_minus_installed_raw_seconds_at_nz: "
            f"{(fastest_aoti_at_nz.seconds_per_call - installed_raw_at_nz.seconds_per_call):.6e}"
        ),
    ]
    markdown_report = _render_markdown_report(
        setup_lines=setup_lines,
        interface_lines=interface_lines,
        decision_lines=decision_lines,
        validation_rows=validation_rows,
        benchmark_rows=benchmark_rows,
    )

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text(markdown_report, encoding="utf-8")
    OUTPUT_JSON_PATH.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote {compiled_package_path}")
    print(f"wrote {OUTPUT_REPORT_PATH}")
    print(f"wrote {OUTPUT_JSON_PATH}")
    print(
        "installed_raw_matches_fresh_raw_on_vulcan_inputs: "
        f"{installed_raw_matches_fresh_raw}"
    )
    print(f"aoti_matches_fresh_raw_on_vulcan_inputs: {aoti_matches_fresh_raw}")
    print(f"fastest_aoti_mode_at_nz: {fastest_aoti_at_nz.model_label}")
    print(f"fastest_model_at_nz: {fastest_overall_at_nz.model_label}")
    print(
        "aoti_minus_installed_raw_seconds_at_nz: "
        f"{(fastest_aoti_at_nz.seconds_per_call - installed_raw_at_nz.seconds_per_call):.6e}"
    )


if __name__ == "__main__":
    main()
