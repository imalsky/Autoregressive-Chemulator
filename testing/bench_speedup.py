#!/usr/bin/env python3
"""Fair ML-vs-mini-chem speed comparison for the Auto-Chem ApJ paper.

Measures amortized per-cell inference latency of the exported 1-step flow map
(export_{cpu,cuda}_dynB_1step_phys.pt2 from RUN_DIR, each timed on its own export
device; the CUDA export only when a GPU is visible) across batch sizes, and reports
the speedup vs the IN-PROCESS mini-chem dlsode per-cell cost (measured separately by
mini_chem/src_mini_chem_dlsode/bench_dlsode.f90) and the crossover batch where the
emulator overtakes the classical solver.

ML inference latency is input-value independent (dense MLP, no data-dependent
branching), so random in-range inputs are valid for timing; only the solver's
cost depends on the physical state (handled by the Fortran driver).

Run:  conda run -n nn python Auto-Chem/testing/bench_speedup.py
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

# Measured mini-chem in-process per-cell cost (us), from bench_dlsode.f90 @ dt=6.4s
# (inherited value; the Fortran benchmark is not part of this repo).
MINICHEM_US_PER_CELL = {"equilibrium": 92.0, "off_equilibrium": 178.0}

# Runtime settings (edit here; no argparse).
# RUN_DIR stays on the v3 run that backs the paper CSV (models/v3/bench_speedup.csv);
# repoint to models/stage2 once the production stage-2 export lands.
RUN_DIR = Path(__file__).resolve().parents[1] / "models" / "v3"
EXPORT_FILES = {"cpu": "export_cpu_dynB_1step_phys.pt2", "cuda": "export_cuda_dynB_1step_phys.pt2"}
WARMUP = 30            # untimed calls per batch size
ITERS = 200            # timed calls per batch size up to ITERS_REF_BATCH ...
ITERS_REF_BATCH = 64   # ... above it the count scales as ITERS * ITERS_REF_BATCH / B ...
MIN_ITERS = 20         # ... but never below this
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _sync(dev: str) -> None:
    """Wait for queued CUDA work so every timed call is complete (no-op on CPU)."""
    if dev == "cuda":
        torch.cuda.synchronize()


def _load_export(path: Path) -> tuple[torch.nn.Module, dict]:
    """Load one .pt2 export: (module on its export device, embedded metadata)."""
    extra = {"metadata.json": ""}
    ep = torch.export.load(str(path), extra_files=extra)
    return ep.module(), json.loads(extra["metadata.json"])


@torch.inference_mode()
def bench_device(model, dev: str, S: int, G: int) -> list[dict]:
    rows: list[dict] = []
    dtype = next(model.parameters()).dtype
    for B in BATCHES:
        try:
            y = torch.rand(B, S, device=dev, dtype=dtype)
            dt = torch.rand(B, device=dev, dtype=dtype)
            g = torch.rand(B, G, device=dev, dtype=dtype)
            for _ in range(WARMUP):
                _ = model(y, dt, g)
            _sync(dev)
            # adaptive iters: fewer for very large batches
            it = max(MIN_ITERS, min(ITERS, ITERS * ITERS_REF_BATCH // max(1, B)))
            samples = []
            for _ in range(it):
                t0 = time.perf_counter()
                _ = model(y, dt, g)
                _sync(dev)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)
            p10, p90 = np.percentile(samples, [10, 90])
            rows.append(
                {
                    "device": dev,
                    "B": B,
                    "call_ms": med * 1e3,
                    "us_per_cell": med * 1e6 / B,
                    "us_per_cell_p10": p10 * 1e6 / B,
                    "us_per_cell_p90": p90 * 1e6 / B,
                    "iters": it,
                }
            )
        except Exception as e:
            print(f"  {dev}: B={B} failed ({type(e).__name__}: {e}); skipping rest of {dev}")
            break
    return rows


def main() -> None:
    print(f"run dir    : {RUN_DIR}")
    print(f"cpu threads: {torch.get_num_threads()}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    print()

    all_rows: list[dict] = []
    for dev, name in EXPORT_FILES.items():
        if dev == "cuda" and not torch.cuda.is_available():
            print("cuda: no GPU visible; skipping the CUDA export")
            continue
        pt2 = RUN_DIR / name
        model, meta = _load_export(pt2)
        S, G = len(meta["species_variables"]), len(meta["global_variables"])
        nparam = sum(p.numel() for p in model.parameters())
        print(f"{dev}: {pt2}  params={nparam:,}  S={S} G={G}")
        all_rows += bench_device(model, dev, S, G)

    # ---- table ----
    print(f"\n{'device':<6} {'B':>5} {'call_ms':>9} {'us/cell':>9} {'p10':>8} {'p90':>8} {'spdup_eq':>9} {'spdup_off':>9}")
    print("-" * 74)
    for r in all_rows:
        s_eq = MINICHEM_US_PER_CELL["equilibrium"] / r["us_per_cell"]
        s_off = MINICHEM_US_PER_CELL["off_equilibrium"] / r["us_per_cell"]
        print(
            f"{r['device']:<6} {r['B']:>5} {r['call_ms']:>9.3f} {r['us_per_cell']:>9.3f} "
            f"{r['us_per_cell_p10']:>8.3f} {r['us_per_cell_p90']:>8.3f} {s_eq:>9.2f} {s_off:>9.2f}"
        )

    # ---- crossover + headline numbers ----
    print("\n=== headline ===")
    for dev in EXPORT_FILES:
        drows = [r for r in all_rows if r["device"] == dev]
        if not drows:
            continue
        b1 = next((r for r in drows if r["B"] == 1), None)
        best = min(drows, key=lambda r: r["us_per_cell"])
        if b1:
            print(f"{dev}: single-cell (B=1) = {b1['call_ms']:.3f} ms/call")
        print(
            f"{dev}: best amortized = {best['us_per_cell']:.3f} us/cell @ B={best['B']} "
            f"-> {MINICHEM_US_PER_CELL['equilibrium']/best['us_per_cell']:.1f}x vs mini-chem(eq), "
            f"{MINICHEM_US_PER_CELL['off_equilibrium']/best['us_per_cell']:.1f}x vs mini-chem(off-eq)"
        )
        for label, base_us in MINICHEM_US_PER_CELL.items():
            cross = next((r for r in drows if r["us_per_cell"] < base_us), None)
            if cross:
                print(f"{dev}: overtakes mini-chem({label}, {base_us:.0f}us) at B>={cross['B']}")
            else:
                print(f"{dev}: never overtakes mini-chem({label}) in tested range")

    # ---- CSV ----
    out = RUN_DIR / "bench_speedup.csv"
    with open(out, "w") as f:
        f.write("device,B,call_ms,us_per_cell,us_per_cell_p10,us_per_cell_p90,iters\n")
        for r in all_rows:
            f.write(
                f"{r['device']},{r['B']},{r['call_ms']:.6f},{r['us_per_cell']:.6f},"
                f"{r['us_per_cell_p10']:.6f},{r['us_per_cell_p90']:.6f},{r['iters']}\n"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
