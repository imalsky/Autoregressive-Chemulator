#!/usr/bin/env bash
# Auto-Chem on daw: preprocess (if needed) -> stage 1 -> stage 2 on ONE GPU with pinned CPUs; after
# each stage its post-training step (gated export, loss curves, rollout plot, rollout-error eval,
# efficiency summary).
#
# Run inside tmux so it survives disconnects:
#   tmux new -s autochem 'bash run_daw.sh all'
# Subcommands: preprocess | stage1 | stage2 | all (default) | smoke
#   stage1 also preprocesses if needed; smoke runs testing/run_local_smoke.sh (synthetic data in
#   smoke_scratch/) in this environment.
#
# Re-running is safe: a finished stage (work_dir/COMPLETED) is not retrained (its post step runs
# again), an interrupted stage resumes from its last.ckpt, and stage 2 starts only from a finished
# stage 1's best.ckpt. Refusals from src/main.py or preprocessing (work_dir in use, a finished or
# moved run, a resume config that differs from the run's, data built differently, another
# preprocessing run) are reported and never retried. Stopping: Ctrl-C, or kill -TERM / -HUP of
# this script's PID, is forwarded as SIGTERM to the running python step (training then stops at
# the end of its current step, exits 143 and resumes from last.ckpt on the next run).
#
# Environment (set here, inherited by every child process): one GPU (CUDA_VISIBLE_DEVICES), THREADS
# CPUs on that GPU's NUMA node (the script pins itself, so preprocessing workers, compile workers,
# the GPU logger and the post steps inherit the pin), OMP/MKL/OPENBLAS/NUMEXPR threads = THREADS,
# TMPDIR (created), inductor/triton caches under ~/.cache, PYTHONPATH=src.
# Optional overrides:
#   AUTOCHEM_GPU        GPU index (default: the first GPU with no compute processes)
#   AUTOCHEM_THREADS    CPU threads to pin and use (default below)
#   AUTOCHEM_GPU_LOG_S  seconds between GPU logger samples (default below)
#   PY                  python interpreter (default: the repo's .venv; a git worktree has none)
# To restrict the CPUs further, start this script under taskset: it picks only CPUs it may use.
#
# Files in logs/ (gitignored): <stage>.log and <stage>_post.log (console, appended),
# <work_dir name>_gpu_util.csv (GPU logger, read by testing/efficiency_summary.py),
# <stage>.resume.json (the last resume config), run_daw.lock.
set -euo pipefail

readonly DEFAULT_THREADS=16          # one job's share of the daw cap below
readonly USER_THREAD_CAP=32          # daw fair-use cap: all of this user's processes together
readonly DEFAULT_GPU_LOG_S=30        # GPU logger cadence; ~3k rows/day
readonly ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PY="${PY:-${ROOT}/.venv/bin/python}"
readonly STAGE1_CFG="${ROOT}/configs/stage1.json"
readonly STAGE2_CFG="${ROOT}/configs/stage2.json"
readonly LOG_DIR="${ROOT}/logs"
readonly THREADS="${AUTOCHEM_THREADS:-${DEFAULT_THREADS}}"
readonly GPU_LOG_S="${AUTOCHEM_GPU_LOG_S:-${DEFAULT_GPU_LOG_S}}"
readonly WHAT="${1:-all}"
readonly POST_TOOLS=(export training_logs predictions eval_rollout_error efficiency_summary)  # testing/<tool>.py, in order

log()  { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2; }  # stderr: pick_gpu's stdout is its result
fail() { log "ERROR: $*"; exit 1; }

case "${WHAT}" in preprocess|stage1|stage2|all|smoke) ;; *) fail "usage: bash run_daw.sh [preprocess|stage1|stage2|all|smoke]";; esac
[ -x "${PY}" ] || fail "missing ${PY}: create the env first (README: uv venv --python 3.12.14 && uv pip sync requirements.lock)"
case "${THREADS}" in ''|*[!0-9]*) fail "AUTOCHEM_THREADS must be a positive integer, got '${THREADS}'";; esac
[ "${THREADS}" -ge 1 ] && [ "${THREADS}" -le "${USER_THREAD_CAP}" ] || fail "AUTOCHEM_THREADS must be in 1..${USER_THREAD_CAP}, got ${THREADS}"
mkdir -p "${LOG_DIR}"

# ---- One training launcher at a time (fail fast; src/main.py and preprocessing also lock their dirs) ----
if [ "${WHAT}" != smoke ]; then
  exec 9>"${LOG_DIR}/run_daw.lock"
  flock -n 9 || fail "another run_daw.sh (or a python step it started) holds ${LOG_DIR}/run_daw.lock"
fi

# ---- GPU choice: AUTOCHEM_GPU, or the first GPU with no compute processes; never a second GPU for this user ----
pick_gpu() {
  local me apps gpus pid uuid idx mine="" target=""
  me="$(id -u)"
  gpus="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)" || return 1
  apps="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)" || return 1
  while IFS=', ' read -r pid uuid; do
    [ -n "${pid}" ] || continue
    if [ "$(ps -o uid= -p "${pid}" 2>/dev/null | tr -d ' ')" = "${me}" ]; then mine+="${uuid}"$'\n'; fi
  done <<<"${apps}"
  if [ -n "${AUTOCHEM_GPU:-}" ]; then
    target="$(awk -F', ' -v i="${AUTOCHEM_GPU}" '$1 == i {print $2}' <<<"${gpus}")"
    [ -n "${target}" ] || { log "ERROR: AUTOCHEM_GPU=${AUTOCHEM_GPU} is not a GPU index"; return 1; }
    if grep -qv -x -e "${target}" -e '' <<<"${mine}"; then
      log "ERROR: your processes already hold another GPU (daw cap: one GPU per user)"; return 1
    fi
    if grep -q "${target}" <<<"${apps}"; then log "WARNING: GPU ${AUTOCHEM_GPU} already has compute processes"; fi
    echo "${AUTOCHEM_GPU}"; return 0
  fi
  if [ -n "${mine}" ]; then
    idx="$(awk -F', ' -v u="$(head -n 1 <<<"${mine}")" '$2 == u {print $1}' <<<"${gpus}")"
    log "ERROR: your processes already hold GPU ${idx} (daw cap: one GPU per user); set AUTOCHEM_GPU=${idx} to share it"
    return 1
  fi
  while IFS=', ' read -r idx uuid; do
    if ! grep -q "${uuid}" <<<"${apps}"; then echo "${idx}"; return 0; fi
  done <<<"${gpus}"
  log "ERROR: no idle GPU (every GPU has compute processes); set AUTOCHEM_GPU to override"
  return 1
}
GPU="$(pick_gpu)" || exit 1

# ---- CPU pinning: THREADS CPUs of the GPU's NUMA node that this process may use; pin this shell ----
bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${GPU}" | tr 'A-F' 'a-f')"
node="$(cat "/sys/bus/pci/devices/${bus#0000}/numa_node" 2>/dev/null || echo -1)"
[ "${node}" -ge 0 ] || node=0
CPUS="$("${PY}" - "${node}" "${THREADS}" <<'PY'
import os, sys
node, n = sys.argv[1], int(sys.argv[2])
cpus = []
for part in open(f"/sys/devices/system/node/node{node}/cpulist").read().strip().split(","):
    lo, _, hi = part.partition("-")
    cpus += range(int(lo), int(hi or lo) + 1)
usable = [c for c in cpus if c in os.sched_getaffinity(0)]
if len(usable) < n:
    sys.exit(f"only {len(usable)} CPUs of NUMA node {node} are usable here; set AUTOCHEM_THREADS <= {len(usable)}")
print(",".join(map(str, usable[-n:])))
PY
)" || fail "no CPU set for ${THREADS} threads on NUMA node ${node}"
taskset -pc "${CPUS}" "$$" >/dev/null || fail "could not pin this shell to CPUs ${CPUS}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}" TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor" TRITON_CACHE_DIR="${HOME}/.cache/triton"
# Compile in-process: inductor's compile-worker pool creates POSIX semaphores on the shared RAM
# disk (seen in the 2026-10-01 pilot), which the daw rules forbid. Cost: slower cold compiles.
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PY
mkdir -p "${TMPDIR}"
log "GPU ${GPU} (NUMA node ${node}); CPUs ${CPUS}; threads ${THREADS}"

if [ "${WHAT}" = smoke ]; then exec bash "${ROOT}/testing/run_local_smoke.sh"; fi

# ---- Python steps: a recorded child (its console appended to logs/<name>.log); signals forwarded to it ----
CHILD_PID=""
GPU_LOGGER_PID=""
POST_FAILURES=()
run_py() {  # run_py <log name> <config> <script>: python's exit status (not tee's)
  local name="$1" cfg="$2" script="$3" rc=0
  # tee -i: Ctrl-C signals the whole process group, and tee must outlive it to log python's stop.
  AUTOCHEM_CONFIG_PATH="${cfg}" "${PY}" -u "${script}" > >(exec tee -i -a "${LOG_DIR}/${name}.log" 9>&-) 2>&1 &
  CHILD_PID=$!
  wait "${CHILD_PID}" || rc=$?
  CHILD_PID=""
  return "${rc}"
}
stop_gpu_logger() {
  if [ -n "${GPU_LOGGER_PID}" ]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
    GPU_LOGGER_PID=""
  fi
}
on_signal() {  # on_signal <signal> <exit status>: SIGTERM to the recorded child only (an async child ignores SIGINT)
  # Forward before any write: stderr may be a hung-up tty or a dead pipe, and a failed write ends this trap.
  if [ -n "${CHILD_PID}" ]; then kill -s TERM "${CHILD_PID}" 2>/dev/null || true; fi
  log "received SIG$1: sent SIGTERM to python pid ${CHILD_PID:-none}; waiting for it" || true
  if [ -n "${CHILD_PID}" ]; then wait "${CHILD_PID}" 2>/dev/null || true; fi
  exit "$2"
}
trap stop_gpu_logger EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

start_gpu_logger() {  # start_gpu_logger <csv>: nvidia-smi sampler of the chosen GPU, appended (never in work_dir)
  [ -s "$1" ] || echo "timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W], clocks.sm [MHz], temperature.gpu, clocks_event_reasons.active" > "$1"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "${GPU}" -l "${GPU_LOG_S}" >> "$1" 9>&- &
  GPU_LOGGER_PID=$!
}

stage_paths() {  # stage_paths <config>: sets PROCESSED, WORK, CKPT exactly as src/main.py resolves them
  local out
  out="$(AUTOCHEM_CONFIG_PATH="$1" "${PY}" - <<'PY'
import shlex
import sys
from pathlib import Path
stdout, sys.stdout = sys.stdout, sys.stderr  # only the assignments below reach the shell
import main  # src/ is on PYTHONPATH
cfg_path = main._resolve_config_path()
cfg = main.resolve_paths(main.load_json_config(cfg_path), cfg_path)
ckpt = cfg["runtime"]["checkpoint"]
values = {
    "PROCESSED": Path(cfg["paths"]["processed_dir"]).expanduser().resolve(),  # as main() does
    "WORK": Path(cfg["paths"]["work_dir"]).expanduser().resolve(),
    "CKPT": "" if ckpt is None else ckpt,
}
for name, value in values.items():
    stdout.write(f"{name}={shlex.quote(str(value))}\n")
PY
)" || fail "could not resolve the paths of $1 with src/main.py"
  eval "${out}"
}

write_resume_cfg() {  # write_resume_cfg <config> <last.ckpt> <out>: absolute paths, checkpoint_mode=resume, runtime.checkpoint
  AUTOCHEM_CONFIG_PATH="$1" "${PY}" - "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path
import main  # src/ is on PYTHONPATH
cfg_path = main._resolve_config_path()
cfg = main.load_json_config(cfg_path)
cfg["paths"] = main.resolve_paths(cfg, cfg_path)["paths"]  # absolute: this file is not next to the config
cfg["training"]["checkpoint_mode"] = "resume"
cfg["runtime"]["checkpoint"] = str(Path(sys.argv[1]).resolve())
Path(sys.argv[2]).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY
}

preprocess() {
  stage_paths "${STAGE1_CFG}"
  if [ -f "${PROCESSED}/normalization.json" ]; then
    # normalization.json is moved in last, so its presence marks a complete build. src/main.py refuses
    # to train on a build whose recorded config or raw files differ: then rebuild it by hand.
    log "processed data present: ${PROCESSED} (skipping preprocessing)"
    return
  fi
  log "preprocessing -> ${PROCESSED}"
  run_py preprocess "${STAGE1_CFG}" "${ROOT}/processing/preprocessing.py" \
    || fail "preprocessing failed with exit status $? (see ${LOG_DIR}/preprocess.log)"
}

run_stage() {  # run_stage <name> <config>: train unless finished, require COMPLETED, then the post step
  local name="$1" cfg="$2" work last resume_cfg tool
  stage_paths "${cfg}"
  work="${WORK}"
  last="${work}/checkpoints/last.ckpt"
  # src/main.py flocks work_dir for its whole life. The -d test is needed: flock on a missing path creates a file.
  if [ -d "${work}" ] && ! flock -n "${work}" true; then fail "${name}: work_dir is in use by another training process: ${work}"; fi
  if [ -f "${work}/COMPLETED" ]; then
    log "${name}: already complete (${work}/COMPLETED); not retraining"
  else
    if [ "${name}" = stage2 ]; then
      [ -f "${CKPT}" ] || fail "stage-1 checkpoint missing: ${CKPT}"
      [ -f "$(dirname "$(dirname "${CKPT}")")/COMPLETED" ] || fail "stage 2 needs a finished stage 1 (no COMPLETED next to ${CKPT})"
    fi
    if [ -f "${last}" ]; then
      resume_cfg="${LOG_DIR}/${name}.resume.json"
      write_resume_cfg "${cfg}" "${last}" "${resume_cfg}" || fail "${name}: could not write ${resume_cfg}"
      log "${name}: resuming from ${last} (config ${resume_cfg})"
      cfg="${resume_cfg}"
    elif [ -d "${work}" ] && [ -n "$(find "${work}" -mindepth 1 -maxdepth 1 ! -name train.log -print -quit)" ]; then
      # A lone train.log (a start that failed before training) counts as empty, as in src/main.py.
      fail "${name}: ${work} holds files but no checkpoints/last.ckpt (crashed before its first checkpoint?). Inspect it (train.log), then remove it and rerun."
    else
      log "${name}: fresh run -> ${work}"
    fi
    start_gpu_logger "${LOG_DIR}/$(basename "${work}")_gpu_util.csv"
    run_py "${name}" "${cfg}" "${ROOT}/src/main.py" || fail "${name}: training exited with status $? (see ${LOG_DIR}/${name}.log)"
    stop_gpu_logger
    [ -f "${work}/COMPLETED" ] || fail "${name}: training exited without writing ${work}/COMPLETED (see ${LOG_DIR}/${name}.log)"
    log "${name}: complete"
  fi
  # Post step: a failure is reported at the end and does not stop the next stage.
  for tool in "${POST_TOOLS[@]}"; do
    log "${name}: testing/${tool}.py"
    if ! run_py "${name}_post" "${cfg}" "${ROOT}/testing/${tool}.py"; then
      log "ERROR: ${name}: testing/${tool}.py failed (see ${LOG_DIR}/${name}_post.log)"
      POST_FAILURES+=("${name}:${tool}")
    fi
  done
}

case "${WHAT}" in
  preprocess) preprocess ;;
  stage1) preprocess; run_stage stage1 "${STAGE1_CFG}" ;;
  stage2) run_stage stage2 "${STAGE2_CFG}" ;;
  all) preprocess; run_stage stage1 "${STAGE1_CFG}"; run_stage stage2 "${STAGE2_CFG}" ;;
esac
[ "${#POST_FAILURES[@]}" -eq 0 ] || fail "post-training steps failed: ${POST_FAILURES[*]}"
log "done: ${WHAT}"
