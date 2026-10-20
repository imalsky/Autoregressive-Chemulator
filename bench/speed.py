"""Inference cost of one model call (plan section R-early (e)): wall time per call and per cell for a batch of cells,
the call being the GCM's step: fp64 normalized state z -> model.delta(z fp32, dt, T/P globals) -> z += delta (fp64).
GPU: CUDA graph of the whole step, TF32 off, 200 timed replays after warm-up. CPU: eager, 1 and 8 threads. Inputs are
random in the normalized ranges (cost does not depend on values). Writes paths.OUT/speed_<run>.json.
    python speed.py <run> [<closed heads' Newton iterations>]
<run>: a run name under paths.RUNS, a run folder or an lhbench spec (ensmed:...; SPEED_NAME then names the output); the
state width is the normalization file's entry count (lhbench: M.S).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import lhbench as LB  # noqa: E402

SPEC = sys.argv[1] if "/" in sys.argv[1] else str(paths.RUNS / sys.argv[1])
RUN = os.environ.get("SPEED_NAME") or Path(SPEC).name   # 2026-10-11: SPEED_NAME names ensmed specs
NEWTON = int(sys.argv[2]) if len(sys.argv) > 2 else None
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
FALLBACK = os.environ.get("SPEED_FALLBACK")  # 2026-10-12: 'lm' times relaxclosedr with the LM rescue on every row
out = {"run": RUN, "closed_iters": NEWTON, "fallback": FALLBACK, "gpu": {}, "cpu1": {}, "cpu8": {}}


def set_fallback(m):
    if FALLBACK is not None:
        for mod in m.modules():
            if isinstance(mod, LB.M.RelaxClosedRHead):
                mod.fallback = FALLBACK
    return m


def inputs(n, dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    z = (torch.rand(n, LB.M.S, generator=g, dtype=torch.float64) * 4 - 2).to(dev)
    gl = torch.rand(n, 2, generator=g).to(dev)
    dtn = torch.rand(n, generator=g).to(dev)
    return z, gl, dtn


dev = torch.device("cuda")
model = set_fallback(LB.load_model(SPEC, "stage2.pt", dev, NEWTON)[0])
print(f"{RUN} on {torch.cuda.get_device_name(0)}; params {sum(p.numel() for p in model.parameters())}", flush=True)
with torch.no_grad():
    for n in (1, 10, 100, 1000, 10_000, 100_000, 730_000):
        z, gl, dtn = inputs(n, dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                z.add_(model.delta(z.float(), dtn, gl).double())
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            z.add_(model.delta(z.float(), dtn, gl).double())
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        reps = 200 if n <= 100_000 else 50
        t = time.perf_counter()
        for _ in range(reps):
            graph.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        out["gpu"][n] = {"s_per_call": dt, "us_per_cell": dt / n * 1e6}
        print(f"  GPU batch {n:7d}: {dt * 1e6:9.1f} us per call, {dt / n * 1e6:9.4f} us per cell", flush=True)
        del graph, z, gl, dtn
        torch.cuda.empty_cache()
cpu_model = set_fallback(LB.load_model(SPEC, "stage2.pt", torch.device("cpu"), NEWTON)[0])
CPU_RUNS = () if os.environ.get("SPEED_GPU_ONLY") else (("cpu1", 1, (1, 10, 100, 1000, 10_000)),
                                                          ("cpu8", 8, (1, 10, 100, 1000, 10_000, 100_000)))  # 2026-10-12
for key, nth, sizes in CPU_RUNS:
    torch.set_num_threads(nth)
    with torch.no_grad():
        for n in sizes:
            z, gl, dtn = inputs(n, torch.device("cpu"))
            for _ in range(3):
                z.add_(cpu_model.delta(z.float(), dtn, gl).double())
            reps = max(3, min(200, int(2e5 / n)))
            t = time.perf_counter()
            for _ in range(reps):
                z.add_(cpu_model.delta(z.float(), dtn, gl).double())
            dt = (time.perf_counter() - t) / reps
            out[key][n] = {"s_per_call": dt, "us_per_cell": dt / n * 1e6}
            print(f"  CPU {nth} thread(s) batch {n:7d}: {dt * 1e6:11.1f} us per call, {dt / n * 1e6:9.3f} us per cell", flush=True)
(paths.OUT / f"speed_{RUN}.json").write_text(json.dumps(out, indent=1))
