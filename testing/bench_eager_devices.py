#!/usr/bin/env python3
"""Eager CPU-vs-MPS latency for the v3 flow map (GPU half of the speed benchmark).

The exported .pt2 is CPU-baked and won't move to MPS, so we time the eager model
(real v3 weights, built via export.py's exact build+load path) on CPU and MPS.
Inference latency is input-value independent (dense MLP), so random normalized
inputs are valid. eager-CPU is cross-checked against the exported-CPU floor
(~70 us/cell) to confirm the eager-MPS number is comparable.

Run:  conda run -n nn python Auto-Chem/testing/bench_eager_devices.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib.util
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MINI = {"eq": 92.0, "off": 178.0}  # measured mini-chem us/cell (single-thread)
CPU_CORES = os.cpu_count() or 1
BATCHES = (1, 8, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768)


def _load_export_module():
    spec = importlib.util.spec_from_file_location("amx", str(ROOT / "testing" / "export.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def bench(base, dev: str) -> list[tuple]:
    m = base.to(dev)
    S, G = int(getattr(m, "S")), int(getattr(m, "G"))
    rows = []
    for B in BATCHES:
        try:
            y = torch.rand(B, S, device=dev)
            dt = torch.rand(B, device=dev)
            g = torch.rand(B, G, device=dev)
            for _ in range(30):
                _ = m.forward_step(y, dt, g)
            _sync(dev)
            it = max(20, min(200, 200 * 64 // max(1, B)))
            ts = []
            for _ in range(it):
                t0 = time.perf_counter()
                _ = m.forward_step(y, dt, g)
                _sync(dev)
                ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            rows.append((dev, B, med * 1e3, med * 1e6 / B))
        except Exception as e:
            print(f"  {dev} B={B} failed ({type(e).__name__}: {e}); stopping {dev}")
            break
    return rows


def main() -> None:
    ex = _load_export_module()
    run_dir = ROOT / "models" / "v3"
    cfg, cfg_path = ex._load_resolved_config(run_dir)
    ckpt = ex._resolve_checkpoint_path("checkpoints/last.ckpt", cfg_path=cfg_path)
    base = ex.create_model(cfg)
    ex._load_weights_strict(base, ckpt)
    base = ex._freeze_for_inference(base)
    nparam = sum(p.numel() for p in base.parameters())
    print(f"params={nparam:,}  S={int(base.S)} G={int(base.G)}  cpu_cores={CPU_CORES} "
          f"torch_threads={torch.get_num_threads()}  mps={torch.backends.mps.is_available()}\n")

    rows = bench(base, "cpu")
    if torch.backends.mps.is_available():
        rows += bench(base, "mps")

    print(f"{'dev':<4}{'B':>8}{'call_ms':>10}{'us/cell':>10}{'sx_eq':>8}{'sx_off':>8}")
    print("-" * 48)
    for d, B, cms, upc in rows:
        print(f"{d:<4}{B:>8}{cms:>10.3f}{upc:>10.3f}{MINI['eq']/upc:>8.2f}{MINI['off']/upc:>8.2f}")

    print("\n=== headline (eager) ===")
    for dev in ("cpu", "mps"):
        dr = [r for r in rows if r[0] == dev]
        if not dr:
            continue
        best = min(dr, key=lambda r: r[3])
        b1 = next((r for r in dr if r[1] == 1), None)
        if b1:
            print(f"{dev}: single-cell {b1[2]:.3f} ms/call")
        print(f"{dev}: best {best[3]:.3f} us/cell @B={best[1]}  "
              f"-> {MINI['eq']/best[3]:.1f}x vs mini-chem(eq), {MINI['off']/best[3]:.1f}x vs off-eq")
    # threading-fair note: mini-chem parallelizes across cells -> divide by cores
    print(f"\nthreading-fair mini-chem on {CPU_CORES} cores: "
          f"~{MINI['eq']/CPU_CORES:.1f} us/cell (eq), ~{MINI['off']/CPU_CORES:.1f} us/cell (off-eq)")


if __name__ == "__main__":
    main()
