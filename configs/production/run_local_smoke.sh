#!/usr/bin/env bash
# End-to-end local smoke of the PRODUCTION configs on synthetic data:
#   synthetic raw -> preprocessing -> stage-1 (2 epochs) -> stage-2 (2 epochs, AR+BPTT,
#   weights_only handoff from stage-1, fourier_dt + loss_discount_gamma wiring).
# Exercises the strict config validation, the new model/trainer code paths, and the
# checkpoint handoff WITHOUT the real dataset. Scratch lives under smoke_scratch/.
#
# Usage:  conda activate nn && bash configs/production/run_local_smoke.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # Auto-Chem repo root
export KMP_DUPLICATE_LIB_OK=TRUE

SCRATCH="smoke_scratch"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"

echo "=== [1/4] synthetic raw ==="
python processing/make_synthetic_raw.py --out-dir "$SCRATCH/raw" --n-files 2 --n-traj 30 --seed 0

echo "=== [2/4] derive smoke configs from production configs ==="
python - "$SCRATCH" <<'EOF'
import json, copy, sys
from pathlib import Path

scratch = sys.argv[1]
root = Path("configs/production")

def smoke_ify(cfg, stage):
    # The smoke config lives inside the scratch dir, so paths are plain names
    # (resolved relative to the config file's directory by src/main.py).
    c = copy.deepcopy(cfg)
    c["paths"]["raw_dir"] = "raw"
    c["paths"]["processed_dir"] = "processed"
    c["paths"]["work_dir"] = f"{stage}_work"
    c["runtime"]["torch_compile"]["enabled"] = False
    c["runtime"]["enable_progress_bar"] = True
    c["preprocessing"]["output_trajectories_per_file"] = 200
    c["preprocessing"]["pool_size"] = 1000
    c["model"]["mlp"]["hidden_dims"] = [64, 64]
    c["dataset"]["windows_per_trajectory"] = 4
    c["training"]["batch_size"] = 64
    c["training"]["max_epochs"] = 2
    c["training"]["scheduler"]["warmup_epochs"] = 1
    c["training"]["early_stopping"]["enabled"] = False
    if stage.startswith("stage2"):
        c["runtime"]["checkpoint"] = "stage1_work/checkpoints/last.ckpt"
        c["training"]["curriculum"]["ramp_epochs"] = 1
    if stage == "stage2_detached":
        # The documented fallback recipe (DECISIONS.md): detached + gamma=0.9.
        ar = c["training"]["autoregressive_training"]
        ar["detach_between_steps"] = True
        ar["backward_per_step"] = True
        ar["loss_discount_gamma"] = 0.9
    return c

for stage, src in [("stage1", "stage1_dt_1e-1_1e3.json"),
                   ("stage2", "stage2_dt_1e-1_1e3.json"),
                   ("stage2_detached", "stage2_dt_1e-1_1e3.json")]:
    cfg = json.load(open(root / src))
    out = Path(scratch) / f"smoke_{stage}.json"
    out.write_text(json.dumps(smoke_ify(cfg, stage), indent=2))
    print("wrote", out)
EOF

echo "=== [3/4] preprocessing + stage-1 (2 epochs) ==="
AUTOCHEM_CONFIG_PATH="$SCRATCH/smoke_stage1.json" python -u processing/preprocessing.py
AUTOCHEM_CONFIG_PATH="$SCRATCH/smoke_stage1.json" python -u src/main.py

echo "=== [4/5] stage-2 default recipe (2 epochs, AR + BPTT + uniform gamma, weights_only handoff) ==="
AUTOCHEM_CONFIG_PATH="$SCRATCH/smoke_stage2.json" python -u src/main.py

echo "=== [5/5] stage-2 FALLBACK recipe (detached + backward_per_step + gamma=0.9) ==="
AUTOCHEM_CONFIG_PATH="$SCRATCH/smoke_stage2_detached.json" python -u src/main.py

echo "=== SMOKE PASSED ==="
ls -la "$SCRATCH/stage1_work/checkpoints" "$SCRATCH/stage2_work/checkpoints" "$SCRATCH/stage2_detached_work/checkpoints"
