#!/usr/bin/env python3
"""run_dir.py - the run directory the testing/ tools act on (one resolver; standard library only).

Same rule as src/main.py: AUTOCHEM_CONFIG_PATH names the training config (required; there is no
default run), and the run directory is that config's paths.work_dir, resolved against the config
file's directory. export.py, aoti_export.py, predictions.py, eval_rollout_error.py,
training_logs.py, bench_speedup.py and efficiency_summary.py all use it, so one config selects
the run (stage 1 or 2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_ENV_VAR = "AUTOCHEM_CONFIG_PATH"


def resolve_run_dir() -> Path:
    """Return the existing run directory named by AUTOCHEM_CONFIG_PATH's paths.work_dir."""
    raw = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{CONFIG_ENV_VAR} is not set. Point it at the run's training config, e.g.:\n"
            f"  {CONFIG_ENV_VAR}=configs/stage2.json python -u testing/export.py"
        )
    cfg_path = Path(raw).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    work = Path(cfg["paths"]["work_dir"]).expanduser()
    run_dir = work if work.is_absolute() else (cfg_path.parent / work).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} (paths.work_dir of {cfg_path}) does not exist")
    return run_dir
