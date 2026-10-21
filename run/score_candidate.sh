#!/bin/bash
# Candidate score: a trained model (A) against the incumbent (B) on the decision set, on GPU1 alone.
#   bash score_candidate.sh [--final] [--check-only] <A name>=<run dir> <B name>=<run dir> <designs dir> <out dir> <CPUs>
# <designs dir> holds decision/ (the decision set: one design per panel, S, L and P plus the readout designs; built by
# bench/score_design.py) and, for --final, the readout designs t2, PD, PH, PX, P2l and P2s. Steps, each skipped when its
# output exists:
#   (1) preflight: every input exists and GPU1 has no compute process; anything missing stops the script here;
#   (2) bench/speed.py on A (32 Newton iterations; Newton only, then the LM rescue on every row) ->
#       <out>/runs/<A>/speed_{keep,lm}.json;
#   (3) --final: the 1000-day run bench/cyc1000.py on A -> <out>/runs/<A>/cyc1000.json;
#   (4) bench/score_run.py of A and B on every decision design -> <out>/runs/<model>/<design>/;
#   (5) bench/score.py A vs B -> <out>/<A>_vs_<B>/decision/;
#   (6) --final: A and B on the readout designs (T2, the old path sets) -> <out>/runs/<model>/<design>/, and on the stored
#       mini-chem yardstick and F1 (yardstick-format designs with their frozen batch files) -> <out>/runs_yardstick/,
#       <out>/runs_f1/ (each with a link to its mini-chem predictions; B's stored yardstick states are linked when they
#       exist), each scored A vs B (and mini-chem where it ran) -> <out>/<A>_vs_<B>/<design>/.
# --check-only runs the preflight alone. Each model runs with its run folder's norm.json (HARNESS_NORM) and LHB_BLOCK 50.
# Marker CANDIDATE-SCORED.
set -u
source "$(dirname "$(readlink -f "$0")")/env.sh"
FINAL=0 CHECK=0
while [ $# -gt 0 ] && [ "${1#--}" != "$1" ]; do
  case $1 in --final) FINAL=1 ;; --check-only) CHECK=1 ;; *) echo "unknown option $1"; exit 2 ;; esac
  shift
done
[ $# = 5 ] && [ "${1#*=}" != "$1" ] && [ "${2#*=}" != "$2" ] || { sed -n '2,3p' "$0"; exit 2; }
A=${1%%=*} RA=$(readlink -m ${1#*=}) B=${2%%=*} RB=$(readlink -m ${2#*=}) DS=$(readlink -m $3) O=$(readlink -m $4) C=$5
[ "$A" != "$B" ] || { echo "the two models need different names"; exit 2; }
YARD=$CHEM_PROJECT/experiments/2026-10-19_minichem-yardstick/_build
F1=$CHEM_PROJECT/experiments/2026-10-20_t0-fix/_build/f1
DEC=$DS/decision
READOUTS="t2 PD PH PX P2l P2s"
log() { echo "$(date '+%F %T') $*"; }

# ---- (1) preflight
miss=0
need() { if [ -e "$1" ]; then log "ok       $1"; else log "MISSING  $1${2:+ ($2)}"; miss=1; fi; }
for r in $RA $RB; do need $r/stage2.pt; need $r/norm.json; done
if [ -d $DEC ]; then
  for p in S L P; do need $DEC/$p/manifest.json "a score panel of the decision set"; done
else
  log "MISSING  $DEC (the decision set does not exist yet; it is built after go)"; miss=1
fi
if [ $FINAL = 1 ]; then
  for d in $READOUTS; do need $DS/$d/manifest.json; done
  need $YARD/design/rows.npz; need $YARD/run/groups.npz; need $YARD/run/minichem
  need $F1/design/rows.npz; need $F1/run/groups.npz "F1's frozen batch file"; need $F1/run/minichem
fi
n=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | grep -c .)
if [ "$n" = 0 ]; then log "ok       GPU1 idle"; else log "BUSY     GPU1 ($n compute processes): speed would be invalid"; miss=1; fi
[ $miss = 0 ] || { log "PREFLIGHT FAILED: nothing was run"; exit 1; }
log "PREFLIGHT OK"
[ $CHECK = 1 ] && exit 0

srun() {  # srun <model> <run dir> <design dir> <out dir> [--groups FILE]
  [ -e $4/meta.json ] && { log "skip score_run $1 on $3 (done)"; return 0; }
  log "score_run $1 on $3"
  LHB_BLOCK=50 HARNESS_NORM=$2/norm.json taskset -c $C $PY $H/bench/score_run.py $2 $3 $4 "${@:5}"
}
sscore() {  # sscore <set> <runs> <out> <models ...>
  [ -e $3/score.json ] && { log "skip score.py on $1 (done)"; return 0; }
  log "score.py on $1"
  mkdir -p $3 && CUDA_VISIBLE_DEVICES= taskset -c $C $PY $H/bench/score.py --set $1 --run $2 --out $3 "${@:4}"
}
stop() { log "FAILED $*"; exit 1; }

# ---- (2) speed, (3) the 1000-day run
mkdir -p $O/runs/$A
for f in keep lm; do
  [ -e $O/runs/$A/speed_$f.json ] && continue
  log "speed $f"
  ( [ $f = lm ] && export SPEED_FALLBACK=lm
    HARNESS_OUT=$O/runs/$A HARNESS_NORM=$RA/norm.json SPEED_GPU_ONLY=1 SPEED_NAME=$f taskset -c $C $PY $H/bench/speed.py $RA 32 ) \
    || stop "speed $f"
done
if [ $FINAL = 1 ] && [ ! -e $O/runs/$A/cyc1000.json ]; then
  log "1000-day run"
  HARNESS_NORM=$RA/norm.json taskset -c $C $PY $H/bench/cyc1000.py $RA $O/runs/$A/cyc1000.json 32 || stop "1000-day run"
fi

# ---- (4) the decision set, (5) its score
for d in $(ls -d $DEC/*/ | sort); do
  d=${d%/}
  [ -e $d/manifest.json ] || continue
  srun $A $RA $d $O/runs/$A/$(basename $d) || stop "score_run $A $d"
  srun $B $RB $d $O/runs/$B/$(basename $d) || stop "score_run $B $d"
done
sscore $DEC $O/runs $O/${A}_vs_$B/decision $A $B || stop "score.py decision"

# ---- (6) final-pick readouts
if [ $FINAL = 1 ]; then
  for d in $READOUTS; do
    srun $A $RA $DS/$d $O/runs/$A/$d || stop "score_run $A $d"
    srun $B $RB $DS/$d $O/runs/$B/$d || stop "score_run $B $d"
    sscore $DS/$d $O/runs $O/${A}_vs_$B/$d $A $B || stop "score.py $d"
  done
  mkdir -p $O/runs_yardstick $O/runs_f1
  [ -e $O/runs_yardstick/minichem ] || ln -s $YARD/run/minichem $O/runs_yardstick/minichem
  [ -e $O/runs_f1/minichem ] || ln -s $F1/run/minichem $O/runs_f1/minichem
  srun $A $RA $YARD/design $O/runs_yardstick/$A --groups $YARD/run/groups.npz || stop "score_run $A yardstick"
  if [ -e $YARD/run/$B/model_states.npz ] && [ ! -e $O/runs_yardstick/$B ]; then
    ln -s $YARD/run/$B $O/runs_yardstick/$B
  else
    srun $B $RB $YARD/design $O/runs_yardstick/$B --groups $YARD/run/groups.npz || stop "score_run $B yardstick"
  fi
  sscore $YARD/design $O/runs_yardstick $O/${A}_vs_$B/yardstick $A $B minichem || stop "score.py yardstick"
  for m in "$A $RA" "$B $RB"; do
    set -- $m
    srun $1 $2 $F1/design $O/runs_f1/$1 --groups $F1/run/groups.npz || stop "score_run $1 F1"
  done
  sscore $F1/design $O/runs_f1 $O/${A}_vs_$B/F1 $A $B minichem || stop "score.py F1"
fi
log "CANDIDATE-SCORED"
