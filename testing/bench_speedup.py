#!/usr/bin/env python3
"""Fair ML-vs-mini-chem speed comparison for the Auto-Chem ApJ paper.

Measures amortized per-cell inference latency of the exported 1-step flow map
across batch sizes on CPU and MPS, and reports the speedup vs the IN-PROCESS
mini-chem dlsode per-cell cost (measured separately by
mini_chem/src_mini_chem_dlsode/bench_dlsode.f90) and the crossover batch where
the emulator overtakes the classical solver.

ML inference latency is input-value independent (dense MLP, no data-dependent
branching), so random in-range inputs are valid for timing; only the solver's
cost depends on the physical state (handled by the Fortran driver).

Run:  conda run -n nn python Auto-Chem/testing/bench_speedup.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import statistics
import time
from pathlib import Path

import torch

# Measured mini-chem in-process per-cell cost (us), from bench_dlsode.f90 @ dt=6.4s.
MINICHEM_US = {"equilibrium": 92.0, "off_equilibrium": 178.0}

DEFAULT_PT2 = (
    Path(__file__).resolve().parents[1] / "models" / "v3" / "export_cpu_dynB_1step_phys.pt2"
)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
S, G = 12, 2  # species, globals (P,T) — fixed for this model family


def _sync(dev: str) -> None:
    if dev == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def bench_device(model, dev: str, warmup: int, iters: int) -> list[dict]:
    rows: list[dict] = []
    dtype = next(model.parameters()).dtype
    for B in BATCHES:
        try:
            y = torch.rand(B, S, device=dev, dtype=dtype)
            dt = torch.rand(B, device=dev, dtype=dtype)
            g = torch.rand(B, G, device=dev, dtype=dtype)
            for _ in range(warmup):
                _ = model(y, dt, g)
            _sync(dev)
            # adaptive iters: fewer for very large batches
            it = max(20, min(iters, iters * 64 // max(1, B)))
            samples = []
            for _ in range(it):
                t0 = time.perf_counter()
                _ = model(y, dt, g)
                _sync(dev)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)
            p10 = sorted(samples)[max(0, int(0.10 * len(samples)) - 1)]
            p90 = sorted(samples)[min(len(samples) - 1, int(0.90 * len(samples)))]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt2", type=str, default=str(DEFAULT_PT2))
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    pt2 = Path(args.pt2)
    print(f"model     : {pt2}")
    base = torch.export.load(str(pt2)).module()
    nparam = sum(p.numel() for p in base.parameters())
    print(f"params    : {nparam:,}")
    print(f"cpu threads: {torch.get_num_threads()}")
    print(f"mps avail : {torch.backends.mps.is_available()}")
    print()

    all_rows: list[dict] = []

    cpu_model = base.to("cpu")
    all_rows += bench_device(cpu_model, "cpu", args.warmup, args.iters)

    if torch.backends.mps.is_available():
        try:
            mps_model = torch.export.load(str(pt2)).module().to("mps")
            all_rows += bench_device(mps_model, "mps", args.warmup, args.iters)
        except Exception as e:
            print(f"  mps: load/move failed ({type(e).__name__}: {e}); cpu only")

    # ---- table ----
    print(f"\n{'device':<6} {'B':>5} {'call_ms':>9} {'us/cell':>9} {'p10':>8} {'p90':>8} {'spdup_eq':>9} {'spdup_off':>9}")
    print("-" * 74)
    for r in all_rows:
        s_eq = MINICHEM_US["equilibrium"] / r["us_per_cell"]
        s_off = MINICHEM_US["off_equilibrium"] / r["us_per_cell"]
        print(
            f"{r['device']:<6} {r['B']:>5} {r['call_ms']:>9.3f} {r['us_per_cell']:>9.3f} "
            f"{r['us_per_cell_p10']:>8.3f} {r['us_per_cell_p90']:>8.3f} {s_eq:>9.2f} {s_off:>9.2f}"
        )

    # ---- crossover + headline numbers ----
    print("\n=== headline ===")
    for dev in ("cpu", "mps"):
        drows = [r for r in all_rows if r["device"] == dev]
        if not drows:
            continue
        b1 = next((r for r in drows if r["B"] == 1), None)
        best = min(drows, key=lambda r: r["us_per_cell"])
        if b1:
            print(f"{dev}: single-cell (B=1) = {b1['call_ms']:.3f} ms/call")
        print(
            f"{dev}: best amortized = {best['us_per_cell']:.3f} us/cell @ B={best['B']} "
            f"-> {MINICHEM_US['equilibrium']/best['us_per_cell']:.1f}x vs mini-chem(eq), "
            f"{MINICHEM_US['off_equilibrium']/best['us_per_cell']:.1f}x vs mini-chem(off-eq)"
        )
        for label, base_us in MINICHEM_US.items():
            cross = next((r for r in drows if r["us_per_cell"] < base_us), None)
            if cross:
                print(f"{dev}: overtakes mini-chem({label}, {base_us:.0f}us) at B>={cross['B']}")
            else:
                print(f"{dev}: never overtakes mini-chem({label}) in tested range")

    # ---- CSV ----
    out = pt2.parent / "bench_speedup.csv"
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
