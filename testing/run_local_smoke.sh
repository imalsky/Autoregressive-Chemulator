#!/usr/bin/env bash
# End-to-end smoke of the PRODUCTION configs on synthetic data: the integration gate.
# Every step asserts its outcome; a failed assertion stops the smoke with "SMOKE CHECK FAILED".
#   1. synthetic raw (the real time grid; one drop_below group, one all-NaN group) -> preprocessing,
#      which must count both rejects and write the train split's raw-trajectory store (one row per
#      kept train group, loadable without pickle)
#   2. stage 1 with the production compile settings, 2 epochs
#   3. a deliberate extension: delete COMPLETED, resume in place with one more epoch (epoch, step and
#      metrics.csv continue; last.ckpt holds the final epoch although no top-k checkpoint saved it)
#   4. a re-run of the finished stage 1, which src/main.py must refuse, leaving best.ckpt unchanged;
#      then, with best.ckpt and COMPLETED deleted (a crash after the final last.ckpt), the same
#      resume must publish both again without training: byte-identical, last.ckpt untouched
#   5. stage 2 default recipe (AR + BPTT, compiled) from stage 1's handoff checkpoint (the file name
#      configs/stage2.json uses, best.ckpt)
#   6. stage 2 fallback recipe (detached + backward_per_step + gamma 0.9), eager
#   7. the Fourier-dt fallback pair (model.mlp.residual=false + model.fourier_dt.enabled=true in both
#      stages), compiled
#   8. export of both stage-2 runs (testing/export.py: CPU + CUDA, with its parity gate); each final
#      .pt2 reloaded: ep.module() without .eval() under inference_mode, finite in-range rows, a NaN row
#      for an out-of-range dt, units and ranges in the metadata, and the export gate re-run against eager
# Every training run: COMPLETED, finite train/val losses at every epoch, the expected epoch count,
# last.ckpt at the final epoch, a ragged last validation batch (as production stage 1 has; stage 2's
# recipe-R validation set is a whole number of batches), and the
# compiled-function variant counts (TORCH_LOGS); a ragged-batch graph must have been compiled and
# stage 2's forward_step may not exceed the latest compile gate's count. Stage-2 live weights must
# differ from the handoff weights and between the two recipes.
# Scratch lives under smoke_scratch/ (gitignored; recreated on every run).
#
# Usage:  bash run_daw.sh smoke     (one pinned GPU and capped threads, as for training)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # Auto-Chem repo root (this script lives in testing/)

PY="${PY:-.venv/bin/python}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|*[!0-9]*) echo "ERROR: CUDA_VISIBLE_DEVICES must name exactly one GPU index; run: bash run_daw.sh smoke" >&2; exit 1 ;;
esac
[ -n "${OMP_NUM_THREADS:-}" ] || { echo "ERROR: OMP_NUM_THREADS is not set; run: bash run_daw.sh smoke" >&2; exit 1; }

SCRATCH="smoke_scratch"
# Stage-2 forward_step graph variants recorded by the latest compile gate (audit T7, 2026-09-30;
# unchanged in the T9 re-run, 2026-10-01: train/no-grad x full/ragged batch, incl. the weights_only
# init validation); a fullgraph compile hard-fails at the 9th variant (recompile limit 8).
GATE_S2_FORWARD_STEP_VARIANTS=4
SMOKE_TORCH_LOGS="recompiles,dynamo"  # dynamo: one "start tracing <fn>" line per compiled variant

rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}/logs"

die() { echo "SMOKE CHECK FAILED: $*" >&2; exit 1; }
# run <label> <config> <script>: one python step; console to the terminal and smoke_scratch/logs/<label>.log
run() { AUTOCHEM_CONFIG_PATH="$2" "${PY}" -u "$3" 2>&1 | tee "${SCRATCH}/logs/$1.log"; }
train() { TORCH_LOGS="${SMOKE_TORCH_LOGS}" run "$1" "${SCRATCH}/smoke_$1.json" src/main.py; }

# Assertions (python, stdlib + torch): check <subcommand> <args...>
CHECKS="$(cat <<'PY'
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def fail(msg):
    sys.exit(f"SMOKE CHECK FAILED: {msg}")


def segment_key(path):
    # metrics.pre_restart.csv, metrics.pre_restart.N.csv (src/main.py), then metrics.csv
    parts = path.name.split(".")
    return (1, 0) if parts[0] == "metrics" and len(parts) == 2 else (0, int(parts[2]) if len(parts) == 4 else 0)


def metrics_by_epoch(work):
    rows = {}
    for path in sorted(list(work.glob("metrics.pre_restart*.csv")) + [work / "metrics.csv"], key=segment_key):
        if not path.is_file():
            fail(f"{path} is missing")
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r["epoch"]:
                    row = rows.setdefault(int(r["epoch"]), {})
                    row.update({k: float(v) for k, v in r.items() if k in ("train_loss", "val_loss") and v})
    return rows


def rejects(summary):
    counts = json.loads(Path(summary).read_text())["group_rejects"]
    if counts["drop_below"] < 1 or counts["non_finite"] < 1:
        fail(f"preprocessing group_rejects {counts}: expected >= 1 drop_below and >= 1 non_finite")
    print(f"OK preprocessing group_rejects {counts}")


def raw_store(processed):
    import numpy as np
    processed = Path(processed)
    summary = json.loads((processed / "preprocessing_summary.json").read_text())
    kept = summary["groups_kept_by_split"]["train"]
    manifest = json.loads((processed / "normalization.json").read_text())
    n_species, n_globals = len(manifest["species_variables"]), len(manifest["global_variables"])
    t_grid = np.load(processed / "raw_trajectories" / "t_grid.npy", allow_pickle=False)
    with np.load(processed / "raw_trajectories" / "train.npz", allow_pickle=False) as z:
        logy, end, g, name = (z[k] for k in ("logy", "end", "globals", "name"))
    n = logy.shape[0]
    if n != kept:
        fail(f"raw_trajectories/train.npz holds {n} groups, train groups_kept {kept}")
    layout = (logy.dtype, end.dtype, g.dtype, name.dtype.kind, logy.shape[1:], end.shape, g.shape, name.shape)
    if layout != (np.float64, np.int16, np.float32, "U", (t_grid.shape[0], n_species), (n,), (n, n_globals), (n,)):
        fail(f"raw_trajectories/train.npz layout {layout} (t_grid {t_grid.shape}, {n_species} species, "
             f"{n_globals} globals)")
    if np.unique(name).size != n:
        fail(f"raw_trajectories/train.npz: {n - np.unique(name).size} repeated group names")
    # The valid rows (t > 0) up to end must hold at least two times (the log-time interpolant).
    lo, hi = int(np.flatnonzero(t_grid > 0)[1]), t_grid.shape[0] - 1
    if end.min() < lo or end.max() > hi:
        fail(f"raw_trajectories/train.npz end rows {end.min()}..{end.max()} outside {lo}..{hi}")
    if not (np.isfinite(logy).all() and np.isfinite(g).all()):
        fail("raw_trajectories/train.npz: non-finite logy or globals")
    print(f"OK raw_trajectories/train.npz: {n} groups = train groups_kept, {t_grid.shape[0]} times x {n_species} "
          f"species, end rows {end.min()}..{end.max()}, unique names, finite")


def trained(work, epochs, log, s2_limit):
    import torch
    work, epochs, text = Path(work), int(epochs), Path(log).read_text(errors="replace")
    done = json.loads((work / "COMPLETED").read_text())
    if done["epochs"] != epochs:
        fail(f"{work}: COMPLETED records {done['epochs']} epochs, expected {epochs}")
    rows = metrics_by_epoch(work)
    if sorted(rows) != list(range(epochs)):
        fail(f"{work}: metrics epochs {sorted(rows)}, expected 0..{epochs - 1}")
    for e, r in sorted(rows.items()):
        if not all(math.isfinite(r.get(k, math.nan)) for k in ("train_loss", "val_loss")):
            fail(f"{work}: epoch {e} train/val loss not finite: {r}")
    last = torch.load(work / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True, mmap=True)
    if last["epoch"] != epochs - 1:
        fail(f"{work}: last.ckpt holds epoch {last['epoch']}, expected the final epoch {epochs - 1}")
    cfg = json.loads((work / "config.resolved.json").read_text())
    batch = cfg["training"]["batch_size"]
    counts = [int(c) for c in re.findall(r"val\(K=\d+\) (\d+) of \d+ enumerated windows", text)]
    if not counts or any(c % batch == 0 for c in counts):
        fail(f"{work}: validation windows {counts} must leave a ragged last batch (batch {batch})")
    if cfg["training"]["checkpoint_mode"] == "weights_only":
        init = json.loads((work / "init_val.json").read_text())
        if not math.isfinite(init["val_loss"]):
            fail(f"{work}: init_val.json val_loss {init['val_loss']}")
    val = [rows[e]["val_loss"] for e in sorted(rows)]
    non_improving = [e for e in range(1, epochs) if val[e] >= min(val[:e])]
    traced = dict(Counter(re.findall(r"torchdynamo start tracing (\S+)", text)))
    tc = cfg["runtime"]["torch_compile"]
    if tc["enabled"]:
        if traced.get("forward_step", 0) < 1:
            fail(f"{work}: compile on but forward_step was never compiled ({traced})")
        ragged = counts[0] % batch
        if not re.search(rf"size mismatch at index 0\. expected {batch}, actual {ragged}\b", text):
            fail(f"{work}: no graph was compiled for the ragged validation batch of {ragged}")
        if cfg["training"]["autoregressive_training"]["enabled"] and traced["forward_step"] > int(s2_limit):
            fail(f"{work}: stage-2 forward_step has {traced['forward_step']} variants, above the compile gate's {s2_limit}")
    print(f"OK {work.name}: {epochs} epochs, val_loss {['%.6g' % v for v in val]}, non-improving epochs "
          f"{non_improving or 'none'}, last.ckpt epoch {last['epoch']}, validation windows {counts} "
          f"(batch {batch}), compile {'on' if tc['enabled'] else 'off'}, variants {traced or 'none'}")


def continued(work, epochs):
    import torch
    work = Path(work)
    steps_per_epoch = json.loads((work / "config.resolved.json").read_text())["dataset"]["train_batches_per_epoch"]
    last = torch.load(work / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True, mmap=True)
    if last["global_step"] != int(epochs) * steps_per_epoch:
        fail(f"{work}: global_step {last['global_step']} after {epochs} epochs of {steps_per_epoch} steps")
    parts = sorted(p.name for p in work.glob("metrics.pre_restart*.csv"))
    if not parts or not (work / "metrics.csv").is_file():
        fail(f"{work}: expected metrics.csv and a metrics.pre_restart part, got {parts}")
    best = json.loads((work / "COMPLETED").read_text())["best"]
    sha = hashlib.sha256((work / "checkpoints" / "best.ckpt").read_bytes()).hexdigest()
    if sha != best["sha256"]:
        fail(f"{work}: best.ckpt sha256 {sha} != COMPLETED best.sha256 {best['sha256']}")
    print(f"OK {work.name} extension: global_step {last['global_step']}, segments {parts} + metrics.csv, "
          f"COMPLETED rewritten (best {Path(best['source']).name})")


def differ(a, a_key, b, b_key):
    import torch
    sa = torch.load(a, map_location="cpu", weights_only=True)[a_key]
    sb = torch.load(b, map_location="cpu", weights_only=True)[b_key]
    keys = sorted(k for k in sa if k.startswith("model."))
    if not keys or keys != sorted(k for k in sb if k.startswith("model.")):
        fail(f"model keys differ between {a}[{a_key}] and {b}[{b_key}]")
    diff = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in keys)
    if diff == 0.0:
        fail(f"{a}[{a_key}] and {b}[{b_key}] hold identical model weights")
    print(f"OK weights differ: {a}[{a_key}] vs {b}[{b_key}], max |diff| {diff:.3e}")


def exported():
    sys.path[:0] = ["testing", "src"]
    import numpy as np
    import torch
    import export as E
    from run_dir import resolve_run_dir
    run_dir = resolve_run_dir()
    for dev in ("cpu", "cuda"):
        path = run_dir / f"export_{dev}_dynB_1step_phys.pt2"
        extra = {"metadata.json": ""}
        ep = torch.export.load(str(path), extra_files=extra)
        meta = json.loads(extra["metadata.json"])
        units = meta["units"]
        if (units["P"], units["T"], units["dt"]) != ("dyn/cm^2 (barye)", "K", "s") or not units.get("species"):
            fail(f"{path}: metadata units {units}")
        if meta["oor_policy"] != "nan_row" or set(meta["oor_box"]) != {"dt_seconds", *meta["global_variables"]}:
            fail(f"{path}: metadata oor_policy/oor_box {meta['oor_policy']} {sorted(meta['oor_box'])}")
        if not 0.0 < meta["dt_min_seconds"] < meta["dt_max_seconds"]:
            fail(f"{path}: metadata dt range {meta['dt_min_seconds']}..{meta['dt_max_seconds']}")
        module = ep.module()  # no .eval(): the deployed call
        rng = np.random.default_rng(0)
        B, S = 8, len(meta["species_variables"])
        y = 10.0 ** rng.uniform(-12.0, -1.0, size=(B, S))
        dt = 10.0 ** rng.uniform(math.log10(meta["dt_min_seconds"]), math.log10(meta["dt_max_seconds"]), size=B)
        cols = []
        for name in meta["global_variables"]:
            box = meta["oor_box"][name]
            u = rng.uniform(box["lo"], box["hi"], size=B)
            cols.append(10.0 ** u if box["coordinate"] == "log10" else u)
        g = np.stack(cols, axis=1)
        dt[0] = 10.0 * meta["dt_max_seconds"]  # out of range: must come back as a NaN row
        bad = E.input_out_of_range_rows(y, dt, g, meta)
        if bad.tolist() != [True] + [False] * (B - 1):
            fail(f"{path}: input out-of-range rows {bad.tolist()}, expected row 0 only")
        with torch.inference_mode():
            out = module(*(torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (y, dt, g))).cpu().numpy()
        if np.isnan(out).all(axis=1).tolist() != bad.tolist() or not np.isfinite(out[~bad]).all():
            fail(f"{path}: NaN rows {np.isnan(out).all(axis=1).tolist()} (expected {bad.tolist()}), in-range rows finite: "
                 f"{bool(np.isfinite(out[~bad]).all())}")
        # The export gate on the final file: eager parity at B = 1, 7, B_MAX, B_MAX + 1 refused, NaN rows.
        _, _, verify = E.build_export(run_dir, E.CHECKPOINT, torch.device(dev), E.EXAMPLE_BATCH)
        verify(module)
        print(f"OK {path.relative_to(run_dir.parent)}: units, ranges, NaN row for dt > dt_max, gate vs eager")


globals()[sys.argv[1]](*sys.argv[2:])
PY
)"
check() { "${PY}" - "$@" <<<"${CHECKS}"; }

echo "=== [1/8] synthetic raw + preprocessing ==="
AUTOCHEM_SYNTH_OUT_DIR="${SCRATCH}/raw" "${PY}" processing/make_synthetic_raw.py
"${PY}" - "${SCRATCH}" <<'EOF'
import copy
import json
import sys
from pathlib import Path

scratch = Path(sys.argv[1])
prod = {s: json.loads(Path(f"configs/{s}.json").read_text()) for s in ("stage1", "stage2")}
HANDOFF = Path(prod["stage2"]["runtime"]["checkpoint"]).name  # the production handoff file (best.ckpt)
BATCH = 64
# Not a multiple of BATCH, so the last validation batch is ragged (production stage 1 has one; stage 2's
# 2621440 = 1280 x 2048 has none, so the smoke is where its ragged-batch graph is exercised).
VAL_SAMPLES = BATCH + BATCH // 4
EXTEND_EPOCHS = 1  # the deliberate extension of the finished stage-1 run


def smoke_ify(stage, *, work, source=None, fourier=False):
    # The smoke config lives inside the scratch dir, so paths are plain names
    # (resolved relative to the config file's directory by src/main.py).
    c = copy.deepcopy(prod[stage])
    c["paths"].update(raw_dir="raw", processed_dir="processed", work_dir=work)
    c["preprocessing"]["output_trajectories_per_file"] = 200
    c["preprocessing"]["pool_size"] = 1000
    c["model"]["mlp"]["hidden_dims"] = [64, 64]
    c["dataset"]["train_batches_per_epoch"] = 12
    c["dataset"]["val_samples"] = VAL_SAMPLES
    c["training"]["batch_size"] = BATCH
    c["training"]["max_epochs"] = 2
    c["training"]["scheduler"]["warmup_epochs"] = 1
    c["training"]["early_stopping"]["enabled"] = False
    if source is not None:  # stage 2: weights_only from the source run's handoff checkpoint
        c["runtime"]["checkpoint"] = f"{source}/checkpoints/{HANDOFF}"
        cur = c["training"]["curriculum"]
        cur["ramp_epochs"] = 1
        # Epoch 0 trains at least two rollout steps past the skipped ones, so the recipes differ.
        cur["start_steps"] = c["training"]["autoregressive_training"]["skip_steps"] + 2
    if fourier:  # the approved fallback pair, flipped together in both stages
        c["model"]["mlp"]["residual"] = False
        c["model"]["fourier_dt"]["enabled"] = True
    return c


cfgs = {"stage1": smoke_ify("stage1", work="stage1_work")}
# Top-k checkpoints every 2 epochs: the extension epoch is then never a top-k save, so last.ckpt
# holding it shows the separate last-checkpoint callback works.
cfgs["stage1"]["runtime"]["checkpointing"]["every_n_epochs"] = 2
ext = copy.deepcopy(cfgs["stage1"])
ext["training"]["checkpoint_mode"] = "resume"
ext["runtime"]["checkpoint"] = "stage1_work/checkpoints/last.ckpt"
ext["training"]["max_epochs"] += EXTEND_EPOCHS
cfgs["stage1_extend"] = ext
cfgs["stage2"] = smoke_ify("stage2", work="stage2_work", source="stage1_work")
det = smoke_ify("stage2", work="stage2_detached_work", source="stage1_work")
# The documented fallback recipe (DECISIONS.md): detached + gamma=0.9, eager (the one uncompiled run).
ar = det["training"]["autoregressive_training"]
ar["detach_between_steps"] = True
ar["backward_per_step"] = True
ar["loss_discount_gamma"] = 0.9
det["runtime"]["torch_compile"]["enabled"] = False
cfgs["stage2_detached"] = det
cfgs["fourier_stage1"] = smoke_ify("stage1", work="fourier_stage1_work", fourier=True)
cfgs["fourier_stage2"] = smoke_ify("stage2", work="fourier_stage2_work", source="fourier_stage1_work", fourier=True)
for name, c in cfgs.items():
    out = scratch / f"smoke_{name}.json"
    out.write_text(json.dumps(c, indent=2))
    print("wrote", out)
EOF
run preprocess "${SCRATCH}/smoke_stage1.json" processing/preprocessing.py
check rejects "${SCRATCH}/processed/preprocessing_summary.json"
check raw_store "${SCRATCH}/processed"

echo "=== [2/8] stage 1 (production compile settings, 2 epochs) ==="
train stage1
check trained "${SCRATCH}/stage1_work" 2 "${SCRATCH}/logs/stage1.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"

echo "=== [3/8] stage 1 deliberate extension (delete COMPLETED, resume with one more epoch) ==="
[ -f "${SCRATCH}/stage1_work/COMPLETED" ] || die "stage 1 wrote no COMPLETED"
rm "${SCRATCH}/stage1_work/COMPLETED"
train stage1_extend
check trained "${SCRATCH}/stage1_work" 3 "${SCRATCH}/logs/stage1_extend.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"
check continued "${SCRATCH}/stage1_work" 3

echo "=== [4/8] re-run of the finished stage 1 (must be refused), then the publish-only resume ==="
before="$(sha256sum "${SCRATCH}/stage1_work/checkpoints/best.ckpt" "${SCRATCH}/stage1_work/COMPLETED")"
if run stage1_rerun "${SCRATCH}/smoke_stage1_extend.json" src/main.py; then die "re-running a finished run was not refused"; fi
grep -q "this run finished" "${SCRATCH}/logs/stage1_rerun.log" || die "the re-run failed for another reason (see ${SCRATCH}/logs/stage1_rerun.log)"
[ "$(sha256sum "${SCRATCH}/stage1_work/checkpoints/best.ckpt" "${SCRATCH}/stage1_work/COMPLETED")" = "${before}" ] \
  || die "the refused re-run changed best.ckpt or COMPLETED"
echo "OK re-run refused; best.ckpt and COMPLETED unchanged"
last_before="$(sha256sum "${SCRATCH}/stage1_work/checkpoints/last.ckpt")"
rm "${SCRATCH}/stage1_work/checkpoints/best.ckpt" "${SCRATCH}/stage1_work/COMPLETED"
run stage1_publish "${SCRATCH}/smoke_stage1_extend.json" src/main.py \
  || die "resuming the final last.ckpt of a run without best.ckpt/COMPLETED failed (see ${SCRATCH}/logs/stage1_publish.log)"
grep -q "publishing it without training" "${SCRATCH}/logs/stage1_publish.log" || die "the publish-only resume did not take the publish path"
if grep -q "epoch [0-9]*: K_train" "${SCRATCH}/logs/stage1_publish.log"; then die "the publish-only resume trained an epoch"; fi
[ "$(sha256sum "${SCRATCH}/stage1_work/checkpoints/best.ckpt" "${SCRATCH}/stage1_work/COMPLETED")" = "${before}" ] \
  || die "the republished best.ckpt or COMPLETED differs from the original"
[ "$(sha256sum "${SCRATCH}/stage1_work/checkpoints/last.ckpt")" = "${last_before}" ] || die "the publish-only resume changed last.ckpt"
echo "OK publish-only resume: best.ckpt and COMPLETED republished byte-identical, no epoch trained"

echo "=== [5/8] stage 2 default recipe (AR + BPTT, compiled, handoff from stage-1 best.ckpt) ==="
train stage2
check trained "${SCRATCH}/stage2_work" 2 "${SCRATCH}/logs/stage2.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"
check differ "${SCRATCH}/stage2_work/checkpoints/last.ckpt" state_dict "${SCRATCH}/stage1_work/checkpoints/best.ckpt" ema_state_dict

echo "=== [6/8] stage 2 fallback recipe (detached + backward_per_step + gamma 0.9, eager) ==="
train stage2_detached
check trained "${SCRATCH}/stage2_detached_work" 2 "${SCRATCH}/logs/stage2_detached.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"
check differ "${SCRATCH}/stage2_work/checkpoints/last.ckpt" state_dict "${SCRATCH}/stage2_detached_work/checkpoints/last.ckpt" state_dict

echo "=== [7/8] Fourier-dt fallback pair (residual=false + fourier_dt, both stages, compiled) ==="
train fourier_stage1
check trained "${SCRATCH}/fourier_stage1_work" 2 "${SCRATCH}/logs/fourier_stage1.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"
train fourier_stage2
check trained "${SCRATCH}/fourier_stage2_work" 2 "${SCRATCH}/logs/fourier_stage2.log" "${GATE_S2_FORWARD_STEP_VARIANTS}"

echo "=== [8/8] export (CPU + CUDA, gated) and reload checks ==="
for name in stage2 fourier_stage2; do
  run "export_${name}" "${SCRATCH}/smoke_${name}.json" testing/export.py
  AUTOCHEM_CONFIG_PATH="${SCRATCH}/smoke_${name}.json" check exported
done

echo "=== SMOKE PASSED ==="
