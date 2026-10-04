#!/usr/bin/env python3
"""efficiency_summary.py - GPU-efficiency summary of one training run (run_daw.sh runs it after each stage).

Reads the run's per-epoch perf metrics (the perf_* columns of metrics.csv and of every
metrics.pre_restart*.csv segment of a resumed run, merged as testing/training_logs.py merges them)
and the launcher's GPU logger, and prints one block: training step time and throughput, achieved
model TFLOP/s, peak GPU memory, the validation share of epoch time, GPU utilization, SM clock,
power, and the achieved TFLOP/s as a fraction of the matmul peak AT THE MEASURED CLOCK.

Why clock-adjusted: an RTX 6000 Ada under sustained matmul load sits at its 300 W power cap and
runs ~1.2-2.2 GHz instead of the 2505 MHz boost that datasheet peaks assume, so the datasheet number
is not an attainable ceiling.

Inputs:
  run directory  paths.work_dir of AUTOCHEM_CONFIG_PATH's config (testing/run_dir.py)
  precision      the run's config.resolved.json: bfloat16 compute -> BF16 peak; float32 -> TF32 peak
                 when precision.allow_tf32 is true and runtime.deterministic is false (src/main.py's
                 rule), otherwise the FP32 CUDA-core peak
  GPU logger     <repo>/logs/<run directory name>_gpu_util.csv: nvidia-smi --query-gpu CSV whose
                 header names utilization.gpu, power.draw, clocks.sm (nvidia-smi's own header writes
                 clocks.current.sm) and clocks_event_reasons.active (columns found by name; values
                 with or without units)
  GPU model      the GPU name in the run's provenance.jsonl, matched to a visible CUDA device for its
                 SM count

Run:
  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u testing/efficiency_summary.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Optional

from run_dir import resolve_run_dir  # testing/ is this script's directory
from training_logs import load_metrics

ROOT = Path(__file__).resolve().parents[1]
GPU_LOG_DIR = ROOT / "logs"
GPU_LOG_SUFFIX = "_gpu_util.csv"

# Dense FLOPs per SM per clock of the matmul unit in use (Ada pro parts): tensor cores for TF32 and
# BF16, CUDA cores for FP32 (128 lanes x 2 per FMA). RTX 6000 Ada: 142 SMs x 2505 MHz x 512 = 182.1
# TFLOPS TF32 and x 1024 = 364.2 TFLOPS BF16, consistent with NVIDIA's 1,457 TFLOPS
# FP8-with-sparsity datasheet figure (2:1 ratios). Measured on an RTX 6000 Ada (2048^3 GEMM against the logged
# SM clock): 435 TF32, 814 BF16 and 160 FP32 FLOP per SM per clock.
FLOP_PER_SM_PER_CLOCK = {"tf32": 512, "bf16": 1024, "fp32": 256}
POWER_CAP_REASON_BIT = 0x4  # clocks_event_reasons: SW power cap (NVML)
# Accepted names of the GPU logger header cells (the text before any " [unit]"): the query name, and
# the canonical name nvidia-smi writes in its own header.
COL_UTIL = ("utilization.gpu",)
COL_POWER = ("power.draw",)
COL_CLOCK = ("clocks.sm", "clocks.current.sm")
COL_REASONS = ("clocks_event_reasons.active",)
MHZ_TO_HZ = 1e6
FLOPS_PER_TFLOP = 1e12


def _finite(values) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _matmul_mode(cfg: dict) -> Optional[str]:
    """'bf16', 'tf32' or 'fp32' for the run's matmuls; None when no peak applies (float64)."""
    compute = str(cfg["precision"]["compute_dtype"]).strip().lower()
    if compute == "bfloat16":
        return "bf16"
    if compute == "float32":
        tf32 = bool(cfg["precision"]["allow_tf32"]) and not bool(cfg["runtime"]["deterministic"])
        return "tf32" if tf32 else "fp32"
    return None


def _gpu_samples(path: Path) -> tuple[list[tuple[float, float, float, bool]], int, int]:
    """(busy samples as (util %, SM MHz, W, power-capped), idle count, unparsed count) of the logger CSV."""
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))
    if not rows:
        raise ValueError(f"empty GPU logger file: {path}")
    header = [h.strip() for h in rows[0]]
    names = [h.split(" [", 1)[0].strip() for h in header]

    def col(accepted: tuple[str, ...]) -> int:
        hits = [i for i, n in enumerate(names) if n in accepted]
        if len(hits) != 1:
            raise ValueError(f"{path}: expected one {' or '.join(accepted)} column in the header {header}")
        return hits[0]

    iu, ic, ip, ir = col(COL_UTIL), col(COL_CLOCK), col(COL_POWER), col(COL_REASONS)
    busy, idle, unparsed = [], 0, 0
    for r in rows[1:]:
        if r and r[0].strip() == header[0]:  # header written again by a restarted logger
            continue
        try:
            u, c, p = (float(r[i].split()[0]) for i in (iu, ic, ip))
            capped = int(r[ir].strip(), 16) & POWER_CAP_REASON_BIT != 0
        except (IndexError, ValueError):  # truncated last line of a stopped logger, "[N/A]" values
            unparsed += 1
            continue
        if u <= 0.0:  # idle samples: startup, preload, compile, between stages
            idle += 1
            continue
        busy.append((u, c, p, capped))
    return busy, idle, unparsed


def _sm_count(run_dir: Path) -> tuple[Optional[int], str]:
    """(SM count, GPU name) of the run's GPU model, or (None, reason)."""
    prov = run_dir / "provenance.jsonl"
    if not prov.is_file():
        return None, f"no {prov.name} in the run directory"
    names = set()
    for line in prov.read_text(encoding="utf-8").splitlines():
        gpu = json.loads(line).get("gpu")
        if isinstance(gpu, dict) and gpu.get("name"):
            names.add(str(gpu["name"]))
    if len(names) != 1:
        return None, f"{prov.name} names {len(names)} GPU models {sorted(names)}"
    name = names.pop()
    import torch  # only for the SM count

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if props.name == name:
            return int(props.multi_processor_count), name
    return None, f"no visible CUDA device is a {name}"


def main() -> None:
    run_dir = resolve_run_dir()
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    mode = _matmul_mode(cfg)
    print(f"== efficiency summary: {run_dir}")
    print(f"   matmul precision: {mode or 'no peak (' + str(cfg['precision']['compute_dtype']) + ')'}")

    m = load_metrics(run_dir)
    step = _finite(m.get("perf_train_step_ms", []))
    tfl: list[float] = []
    if step:
        sps = _finite(m["perf_train_samples_per_s"])
        tfl = _finite(m["perf_train_tflops"])
        print(f"   epochs with perf metrics: {len(step)} (epoch 0 includes compile warm-up; medians are robust to it)")
        print(f"   train step: median {statistics.median(step):.2f} ms | samples/s median {statistics.median(sps):,.0f}")
        print(f"   achieved model TFLOP/s: median {statistics.median(tfl):.3g} (min {min(tfl):.3g}, max {max(tfl):.3g})")
        mem = _finite(m.get("perf_peak_mem_gb", []))
        if mem:
            print(f"   peak GPU memory: {max(mem):.2f} GB")
        if "perf_val_time_s" in m:
            pairs = [(v, e) for v, e in zip(m["perf_val_time_s"], m["epoch_time_sec"], strict=True) if math.isfinite(v) and math.isfinite(e)]
            if pairs:
                print(f"   validation share of epoch time: {100 * sum(v for v, _ in pairs) / sum(e for _, e in pairs):.1f}%")
    else:
        print("   no per-epoch perf metrics (perf_train_step_ms) in the metrics files")

    gpu_log = GPU_LOG_DIR / f"{run_dir.name}{GPU_LOG_SUFFIX}"
    if not gpu_log.is_file():
        print(f"   no GPU logger file at {gpu_log}")
        return
    busy, idle, unparsed = _gpu_samples(gpu_log)
    print(f"   GPU logger {gpu_log.name}: {len(busy)} busy samples ({idle} idle, {unparsed} unparsed)")
    if not busy:
        return
    mean_clk = statistics.mean(c for _, c, _, _ in busy)
    capped = sum(1 for *_, cap in busy if cap)
    print(f"   utilization mean {statistics.mean(u for u, *_ in busy):.0f}% | SM clock mean {mean_clk:.0f} MHz"
          f" | power mean {statistics.mean(p for _, _, p, _ in busy):.0f} W | power-capped {100 * capped / len(busy):.0f}% of samples")
    if not tfl or mode is None:
        return
    sms, gpu_name = _sm_count(run_dir)
    if sms is None:
        print(f"   clock-adjusted peak unavailable: {gpu_name}")
        return
    peak = sms * FLOP_PER_SM_PER_CLOCK[mode] * mean_clk * MHZ_TO_HZ / FLOPS_PER_TFLOP
    print(f"   {mode.upper()} peak at the mean clock ({gpu_name}, {sms} SMs): {peak:.0f} TFLOP/s -> achieved "
          f"{100 * statistics.median(tfl) / peak:.0f}% of it")


if __name__ == "__main__":
    main()
