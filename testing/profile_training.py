#!/usr/bin/env python3
"""profile_training.py - profile a few training steps of the real entrypoint (src/main.py).

Runs src/main.main() unmodified on a copy of AUTOCHEM_CONFIG_PATH's config and forces a short fit
(one epoch of WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS training batches and VAL_BATCHES
validation batches) with a probe callback that
  1) times TIMED_STEPS steps between two device synchronizes (no probe sync inside the window),
  2) runs torch.profiler over the next PROFILED_STEPS steps (its overhead stays out of 1).

The copy differs from the config only in paths.work_dir, runtime.checkpointing.enabled = false and
training.early_stopping.enabled = false, so the run directory the config names is never touched.
Profile a fresh (checkpoint_mode none) or a weights_only config; a resume continues an existing run
directory and is refused, and so is a dataset.train_batches_per_epoch below the profiled steps.

Outputs, in <paths.work_dir>_profile/ beside the config's run directory (refused if it exists):
  config.json           the derived config
  work/                 the 1-epoch run main.py writes (train.log, metrics.csv, COMPLETED, ...)
  profile_summary.json  timed step time, samples/s, model TFLOP/s (src/trainer.py's FLOP count:
                        no-grad skip steps forward-only), peak memory, GPU busy fraction of the
                        profiled steps, per-step sync/copy/launch counts, top device and CPU ops
  profile_trace.json    Chrome trace of the profiled steps (open in ui.perfetto.dev)
The torch.profiler key_averages table of the profiled steps is printed.

Run:
  AUTOCHEM_CONFIG_PATH=configs/stage1.json python -u testing/profile_training.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightning.pytorch as pl  # noqa: E402
import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import main as entry  # noqa: E402  (src/main.py)
from run_dir import CONFIG_ENV_VAR  # noqa: E402  (testing/ is this script's directory)
from utils import load_json_config  # noqa: E402

# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------
WARMUP_STEPS = 15     # compile, CUDA-graph recording and allocator warm-up
TIMED_STEPS = 60      # timed between two synchronizes
PROFILED_STEPS = 10   # under torch.profiler
VAL_BATCHES = 2
TOP_OPS = 12          # rows of the top-op lists (the printed table shows twice as many)
OP_NAME_CHARS = 90    # op names are truncated to this length in the summary

PROFILE_DIR_SUFFIX = "_profile"
SUMMARY_NAME = "profile_summary.json"
TRACE_NAME = "profile_trace.json"
US_PER_S = 1e6
US_PER_MS = 1e3
MS_PER_S = 1e3
BYTES_PER_GB = 1e9
FLOPS_PER_TFLOP = 1e12


def _derive_config() -> Path:
    """Write the profiling copy of AUTOCHEM_CONFIG_PATH's config; return its path."""
    raw = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{CONFIG_ENV_VAR} is not set. Point it at a training config, e.g.:\n"
            f"  {CONFIG_ENV_VAR}=configs/stage1.json python -u testing/profile_training.py"
        )
    src = Path(raw).expanduser().resolve()
    cfg = entry.resolve_paths(load_json_config(src), src)  # absolute paths, so the copy can move
    if str(cfg["training"]["checkpoint_mode"]).lower() == "resume":
        raise ValueError("profile a fresh or weights_only config: a resume continues an existing run directory")
    batches = int(cfg["dataset"]["train_batches_per_epoch"])
    if batches < WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS:
        raise ValueError(
            f"dataset.train_batches_per_epoch={batches} < the {WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS} "
            "profiled training steps; raise it in the profiled config"
        )
    work = Path(cfg["paths"]["work_dir"])
    out_dir = work.parent / f"{work.name}{PROFILE_DIR_SUFFIX}"
    if out_dir.exists():
        raise FileExistsError(f"profile directory exists (remove it to profile again): {out_dir}")
    out_dir.mkdir(parents=True)
    cfg["paths"]["work_dir"] = str(out_dir / "work")
    cfg["runtime"]["checkpointing"]["enabled"] = False
    cfg["training"]["early_stopping"]["enabled"] = False
    path = out_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


class _Probe(pl.Callback):
    """Times TIMED_STEPS steps, then profiles PROFILED_STEPS steps, and writes the outputs."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.summary: Dict[str, Any] = {}
        self.done = False
        self._prof: Any = None

    @staticmethod
    def _cuda(pl_module: pl.LightningModule) -> bool:
        return pl_module.device.type == "cuda"

    def _sync(self, pl_module: pl.LightningModule) -> None:
        if self._cuda(pl_module):
            torch.cuda.synchronize(pl_module.device)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if batch_idx == WARMUP_STEPS:
            self._sync(pl_module)
            if self._cuda(pl_module):
                torch.cuda.reset_peak_memory_stats(pl_module.device)
            self._t0 = time.perf_counter()
        elif batch_idx == WARMUP_STEPS + TIMED_STEPS:
            self._sync(pl_module)
            self.summary["timed_wall_s"] = time.perf_counter() - self._t0
            activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self._cuda(pl_module) else [])
            self._prof = profile(activities=activities)
            self._prof.__enter__()
            self._p0 = time.perf_counter()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: Any, _outputs: Any, batch: Any, batch_idx: int) -> None:
        if batch_idx != WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS - 1:
            return
        self._sync(pl_module)
        wall = time.perf_counter() - self._p0
        self._prof.__exit__(None, None, None)
        cuda = self._cuda(pl_module)
        r = self.summary
        batch_size = int(batch["y"].shape[0])
        flops_per_step = batch_size * pl_module.train_forwards_per_sample() * pl_module.fwd_flops_per_sample_step
        step_s = r["timed_wall_s"] / TIMED_STEPS
        r.update(
            device=str(pl_module.device),
            batch=batch_size,
            warmup_steps=WARMUP_STEPS,
            timed_steps=TIMED_STEPS,
            step_ms=MS_PER_S * step_s,
            samples_per_s=batch_size / step_s,
            fwd_flops_per_sample_step=pl_module.fwd_flops_per_sample_step,
            forwards_per_sample=pl_module.train_forwards_per_sample(),
            train_flops_per_step=flops_per_step,
            achieved_tflops=flops_per_step / step_s / FLOPS_PER_TFLOP,
            peak_mem_gb=torch.cuda.max_memory_allocated(pl_module.device) / BYTES_PER_GB if cuda else None,
            profiled_steps=PROFILED_STEPS,
            profiled_wall_s=wall,
        )
        ka = self._prof.key_averages()
        if cuda:
            dev_events = [e for e in self._prof.events() if e.device_type == DeviceType.CUDA]
            # Union of the device-event intervals (us): the fraction of wall time the GPU had work.
            busy, cur_s, cur_e = 0.0, None, None
            for s0, e0 in sorted((e.time_range.start, e.time_range.end) for e in dev_events):
                if cur_e is None or s0 > cur_e:
                    if cur_e is not None:
                        busy += cur_e - cur_s
                    cur_s, cur_e = s0, e0
                else:
                    cur_e = max(cur_e, e0)
            if cur_e is not None:
                busy += cur_e - cur_s
            r["gpu_busy_frac_profiled"] = busy / (wall * US_PER_S)
            r["gpu_kernel_ms_per_step"] = sum(e.time_range.elapsed_us() for e in dev_events) / US_PER_MS / PROFILED_STEPS

        def per_step(pred: Any) -> float:
            return sum(e.count for e in ka if pred(e.key)) / PROFILED_STEPS

        r["per_step"] = {
            "cudaStreamSynchronize": per_step(lambda k: k == "cudaStreamSynchronize"),
            "cudaDeviceSynchronize": per_step(lambda k: k == "cudaDeviceSynchronize"),
            "cudaMemcpyAsync": per_step(lambda k: k == "cudaMemcpyAsync"),
            "Memcpy HtoD events": per_step(lambda k: "Memcpy HtoD" in k),
            "kernel launches": per_step(lambda k: k in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")),
            "cudaGraphLaunch": per_step(lambda k: k == "cudaGraphLaunch"),
            "aten::item": per_step(lambda k: k == "aten::item"),
            "aten::_local_scalar_dense": per_step(lambda k: k == "aten::_local_scalar_dense"),
        }
        loader = [e for e in ka if "DataLoader" in e.key and "__next__" in e.key]
        r["dataloader_next_cpu_ms_per_step"] = sum(e.cpu_time_total for e in loader) / US_PER_MS / PROFILED_STEPS
        if cuda:
            top_dev = sorted((e for e in ka if e.device_type == DeviceType.CUDA), key=lambda e: e.self_device_time_total, reverse=True)
            r["top_device_ops_ms_per_step"] = [
                (e.key[:OP_NAME_CHARS], round(e.self_device_time_total / US_PER_MS / PROFILED_STEPS, 3)) for e in top_dev[:TOP_OPS]
            ]
        top_cpu: List[Any] = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:TOP_OPS]
        r["top_self_cpu_ops_ms_per_step"] = [
            (e.key[:OP_NAME_CHARS], round(e.self_cpu_time_total / US_PER_MS / PROFILED_STEPS, 3)) for e in top_cpu
        ]
        sort_by = "self_device_time_total" if cuda else "self_cpu_time_total"
        print(ka.table(sort_by=sort_by, row_limit=2 * TOP_OPS))
        self._prof.export_chrome_trace(str(self.out_dir / TRACE_NAME))
        (self.out_dir / SUMMARY_NAME).write_text(json.dumps(r, indent=2), encoding="utf-8")
        self.done = True
        headline = {k: r[k] for k in ("step_ms", "samples_per_s", "achieved_tflops", "peak_mem_gb")}
        print(f"PROFILE_DONE {json.dumps(headline)} -> {self.out_dir}")


def main() -> None:
    cfg_path = _derive_config()
    probe = _Probe(cfg_path.parent)
    orig_init = pl.Trainer.__init__

    def init(self: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        steps = WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS
        kwargs.update(
            max_epochs=1,
            limit_train_batches=steps,
            limit_val_batches=VAL_BATCHES,
            # Nothing is logged per step; this only avoids Lightning's interval-longer-than-epoch warning.
            log_every_n_steps=min(int(kwargs.get("log_every_n_steps", steps)), steps),
        )
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [probe]
        orig_init(self, *args, **kwargs)

    pl.Trainer.__init__ = init
    os.environ[CONFIG_ENV_VAR] = str(cfg_path)
    entry.main()
    if not probe.done:
        raise RuntimeError(f"the fit ended before {WARMUP_STEPS + TIMED_STEPS + PROFILED_STEPS} training steps")


if __name__ == "__main__":
    main()
