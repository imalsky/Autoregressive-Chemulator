#!/usr/bin/env python3
"""ML-vs-mini-chem speed comparison for the Auto-Chem ApJ paper (diagnostic).

Measures the per-cell inference latency of the run's exported 1-step model
(export_{cpu,cuda}_dynB_1step_dlog10.pt2 in the run directory named by AUTOCHEM_CONFIG_PATH,
testing/run_dir.py; each timed on its own export device, the CUDA export only when a GPU is
visible) across batch sizes and CPU thread counts, and compares it with the IN-PROCESS mini-chem
dlsode per-cell cost (MINICHEM_US_PER_CELL, measured separately by
mini_chem/src_mini_chem_dlsode/bench_dlsode.f90; not part of this repo). The export returns a
per-row log10 increment; the timed call is the module alone, without the host's update of its
state (testing/export.py host_step: a few elementwise operations per cell).

Inputs are drawn inside the export's trained box (species log range, dt range, P/T box from the
metadata; seeded), so no row is NaN-marked. ML latency does not depend on the input values (dense
MLP, no data-dependent branching); only the solver's cost depends on the physical state.

Timing: every (device, threads) pass times each batch size in BATCHES and DEPLOY_BATCH_SIZES,
interleaved over ROUNDS rounds; the reported cost is the median over rounds of each round's
median call time. The headline per (device, threads) is the per-cell cost at each DEPLOY batch
size; the mini-chem ratio is printed only where threads == MINICHEM_THREADS (mini-chem's own
thread count) and is cross-host until mini-chem is timed on this machine. The lowest per-cell
cost over the sweep is peak throughput, not the per-call cost at the host batch. A batch outside
the export's dynamic_batch range is skipped with a message; any other failure stops the run with a
non-zero exit before the CSV is written.

Run:  AUTOCHEM_CONFIG_PATH=configs/stage2.json python -u testing/bench_speedup.py
Writes <run_dir>/bench_speedup.csv (leading '#' provenance lines).
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from run_dir import resolve_run_dir  # testing/ is this script's directory

# Measured mini-chem in-process per-cell cost (us), from bench_dlsode.f90 @ dt=6.4s
# (inherited value; the Fortran benchmark is not part of this repo).
MINICHEM_US_PER_CELL = {"equilibrium": 92.0, "off_equilibrium": 178.0}
# mini-chem's dlsode driver is single-threaded (inherited), so ratios are printed only at 1 thread.
MINICHEM_THREADS = 1
# The machine the mini-chem numbers were timed on: unrecorded (the author supplies it); until then
# every ratio is cross-host.
MINICHEM_HOST = "unrecorded (author to supply)"

EXPORT_FILES = {"cpu": "export_cpu_dynB_1step_dlog10.pt2", "cuda": "export_cuda_dynB_1step_dlog10.pt2"}
# The metadata format these files must carry (testing/export.py EXPORT_FORMAT; this script stays standalone).
EXPORT_FORMAT = "1step_dlog10_dynB"
WARMUP = 30            # untimed calls per batch size, after every thread-count change
ITERS = 200            # timed calls per batch size up to ITERS_REF_BATCH ...
ITERS_REF_BATCH = 64   # ... above it the count scales as ITERS * ITERS_REF_BATCH / B ...
MIN_ITERS = 20         # ... but never below this
ROUNDS = 5             # interleaved rounds over the batch sizes; the median over rounds is reported
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
# Host batch per call. Inherited: the VULCAN HD189 example nz (cfg_examples/vulcan_cfg_HD189.py:69),
# unconfirmed for the VULCAN-0D reference state; replace with the author-confirmed nz and the Canoe
# per-call cell count.
DEPLOY_BATCH_SIZES = (150,)
BENCH_INPUT_SEED = 0
OUT_NAME = "bench_speedup.csv"
_CPUINFO_PATH = Path("/proc/cpuinfo")


def _sync(dev: str) -> None:
    """Wait for queued CUDA work so every timed call is complete (no-op on CPU)."""
    if dev == "cuda":
        torch.cuda.synchronize()


def _load_export(path: Path) -> tuple[torch.nn.Module, dict]:
    """Load one .pt2 export: (module on its export device, embedded metadata); refuses another format."""
    extra = {"metadata.json": ""}
    ep = torch.export.load(str(path), extra_files=extra)
    meta = json.loads(extra["metadata.json"])
    if meta.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path}: metadata format {meta.get('format')!r}, expected {EXPORT_FORMAT!r}; "
                         "re-export the run with testing/export.py")
    return ep.module(), meta


def _in_box_inputs(meta: dict, B: int, dev: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """B rows inside the export's trained box (seeded): species log-uniform in [log_min, log_max],
    dt log-uniform in the dt range, P and T uniform in their oor_box coordinate."""
    rng = np.random.default_rng(BENCH_INPUT_SEED)
    st = meta["per_key_stats"]
    lo = np.array([st[s]["log_min"] for s in meta["species_variables"]])
    hi = np.array([st[s]["log_max"] for s in meta["species_variables"]])
    y = 10.0 ** (lo + rng.random((B, lo.size)) * (hi - lo))
    box = meta["oor_box"]

    def draw(name: str) -> np.ndarray:
        b = box[name]
        v = b["lo"] + rng.random(B) * (b["hi"] - b["lo"])
        return 10.0 ** v if b["coordinate"] == "log10" else v

    dt = draw("dt_seconds")
    g = np.column_stack([draw(name) for name in meta["global_variables"]])
    as_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)  # noqa: E731
    return as_t(y), as_t(dt), as_t(g)


@torch.inference_mode()
def _time_batch(model, dev: str, meta: dict, B: int) -> list[float]:
    """Warm up, then return per-call seconds for one batch size."""
    y, dt, g = _in_box_inputs(meta, B, dev)
    for _ in range(WARMUP):
        _ = model(y, dt, g)
    _sync(dev)
    it = max(MIN_ITERS, min(ITERS, ITERS * ITERS_REF_BATCH // max(1, B)))
    samples = []
    for _ in range(it):
        t0 = time.perf_counter()
        out = model(y, dt, g)
        _sync(dev)
        samples.append(time.perf_counter() - t0)
    if bool(torch.isnan(out).any()):
        raise RuntimeError(f"{dev}: B={B} produced NaN rows for in-box inputs")
    return samples


def bench_pass(model, dev: str, meta: dict, threads: int) -> list[dict]:
    """One (device, threads) pass: ROUNDS interleaved rounds over every batch size in range."""
    b_min, b_max = int(meta["dynamic_batch"]["min"]), int(meta["dynamic_batch"]["max"])
    sizes = []
    for B in sorted(set(BATCHES) | set(DEPLOY_BATCH_SIZES)):
        if not b_min <= B <= b_max:
            print(f"  {dev}/{threads}t: B={B} outside the export's dynamic_batch [{b_min}, {b_max}]; skipped")
            continue
        sizes.append(B)
    per_round: dict[int, list[float]] = {B: [] for B in sizes}
    for _ in range(ROUNDS):
        for B in sizes:
            per_round[B].append(statistics.median(_time_batch(model, dev, meta, B)))
    rows = []
    for B in sizes:
        med = statistics.median(per_round[B])
        rows.append({
            "device": dev, "threads": threads, "B": B, "call_ms": med * 1e3, "us_per_cell": med * 1e6 / B,
            "us_per_cell_round_min": min(per_round[B]) * 1e6 / B, "us_per_cell_round_max": max(per_round[B]) * 1e6 / B,
            "deploy": B in DEPLOY_BATCH_SIZES,
        })
    return rows


def _cpu_model() -> str | None:
    if not _CPUINFO_PATH.exists():
        return None
    for line in _CPUINFO_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    run_dir = resolve_run_dir()
    startup_threads = torch.get_num_threads()
    affinity = len(os.sched_getaffinity(0))
    threads = tuple(dict.fromkeys((MINICHEM_THREADS, startup_threads)))
    if any(t > affinity for t in threads):
        raise RuntimeError(f"thread counts {threads} exceed the {affinity} CPUs this process may run on")
    load_start = os.getloadavg()
    print(f"run dir    : {run_dir}")
    print(f"cpu threads: {threads} (startup {startup_threads}, affinity {affinity}); cpu {_cpu_model()}")
    print(f"cuda avail : {torch.cuda.is_available()}")

    all_rows: list[dict] = []
    metas: dict[str, dict] = {}
    for dev, name in EXPORT_FILES.items():
        if dev == "cuda" and not torch.cuda.is_available():
            print("cuda: no GPU visible; skipping the CUDA export")
            continue
        pt2 = run_dir / name
        model, meta = _load_export(pt2)
        metas[dev] = meta
        print(f"{dev}: {pt2}  params={sum(p.numel() for p in model.parameters()):,}  "
              f"S={len(meta['species_variables'])} G={len(meta['global_variables'])}")
        # CPU kernels follow the thread count; the CUDA export is timed once (startup threads).
        for t in (threads if dev == "cpu" else (startup_threads,)):
            torch.set_num_threads(t)
            all_rows += bench_pass(model, dev, meta, t)
    torch.set_num_threads(startup_threads)
    load_end = os.getloadavg()

    print(f"\n{'device':<6} {'thr':>4} {'B':>5} {'call_ms':>9} {'us/cell':>9} {'rnd_min':>8} {'rnd_max':>8}")
    for r in all_rows:
        print(f"{r['device']:<6} {r['threads']:>4} {r['B']:>5} {r['call_ms']:>9.3f} {r['us_per_cell']:>9.3f} "
              f"{r['us_per_cell_round_min']:>8.3f} {r['us_per_cell_round_max']:>8.3f}" + ("  <- deploy" if r["deploy"] else ""))

    print("\n=== headline per (device, threads) ===")
    for dev, t in dict.fromkeys((r["device"], r["threads"]) for r in all_rows):
        rows = [r for r in all_rows if r["device"] == dev and r["threads"] == t]
        for r in (r for r in rows if r["deploy"]):
            line = f"{dev}/{t}t: {r['us_per_cell']:.3f} us/cell at the deploy batch B={r['B']}"
            if t == MINICHEM_THREADS:
                line += " -> " + ", ".join(f"{base / r['us_per_cell']:.1f}x vs mini-chem({lab})"
                                           for lab, base in MINICHEM_US_PER_CELL.items())
                line += f" (cross-host: mini-chem timed on {MINICHEM_HOST})"
            print(line)
        best = min(rows, key=lambda r: r["us_per_cell"])
        print(f"{dev}/{t}t: peak throughput (not the per-call cost at the host batch) "
              f"{best['us_per_cell']:.3f} us/cell at B={best['B']}")
        if t == MINICHEM_THREADS:
            for lab, base in MINICHEM_US_PER_CELL.items():
                below = [r["B"] for r in rows if r["us_per_cell"] < base]
                print(f"{dev}/{t}t: timed B with per-cell cost below mini-chem({lab}, {base:.0f} us): {below or 'none'}")

    out = run_dir / OUT_NAME
    with open(out, "w") as f:
        for dev, meta in metas.items():
            f.write(f"# {dev}: export={run_dir / EXPORT_FILES[dev]} run_dir={meta.get('run_dir')} "
                    f"checkpoint={meta.get('checkpoint_path')} checkpoint_sha256={meta.get('checkpoint_sha256')} "
                    f"torch={meta.get('torch_version')}\n")
        f.write(f"# host: cpu={_cpu_model()} affinity={affinity} startup_threads={startup_threads} torch={torch.__version__} "
                f"loadavg_start={load_start} loadavg_end={load_end} rounds={ROUNDS} "
                f"minichem_threads={MINICHEM_THREADS} minichem_host={MINICHEM_HOST}\n")
        f.write("device,threads,B,call_ms,us_per_cell,us_per_cell_round_min,us_per_cell_round_max,deploy\n")
        for r in all_rows:
            f.write(f"{r['device']},{r['threads']},{r['B']},{r['call_ms']:.6f},{r['us_per_cell']:.6f},"
                    f"{r['us_per_cell_round_min']:.6f},{r['us_per_cell_round_max']:.6f},{int(r['deploy'])}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
