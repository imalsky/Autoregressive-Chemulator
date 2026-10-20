#!/bin/bash
# Candidate bench and judge (template of the scale phase's p/cbgp.sh with the run as an argument; GPU1 alone, checked first):
# (1) idle-GPU speed (bench/speed.py, 32 Newton iterations; Newton only, then the LM rescue on every row) -> paths.OUT/
#     speed_<label>_idle_{keep,lm}.json; (2) the model-only 1000-day run (bench/cyc1000.py) -> paths.OUT/cyc1000_<label>.json;
# (3) every development set (run/bench_dev.sh: PD PX P2l P2s PH DEV6 T2 SD0.1 SD1), one bench at a time; (4) bench/judge.py
# <label> vs <reference label> -> paths.OUT/judge_<label>.txt. Each step is skipped when its output exists. label = the run
# folder's name. Markers CANDIDATE-BENCHED, CANDIDATE-DONE.
#   setsid nohup bash bench_candidate.sh <run dir> <reference label> <CPUs> > <log> 2>&1 &
source "$(dirname "$(readlink -f "$0")")/env.sh"
RUN=$(readlink -f $1) REF=$2 C=$3
L=$(basename $RUN)
OUT=$($PY -c "import sys; sys.path.insert(0, '$H/lib'); import paths; print(paths.OUT)")
NORM=$CHEM_SCRATCH/arch_gcm17m_res/norm.json
log() { echo "$(date '+%F %T') $*"; }
n=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$n" = 0 ] || { log "GPU1 not idle ($n processes): speed would be invalid; stop"; exit 1; }
for f in keep lm; do
  [ -e $OUT/speed_${L}_idle_$f.json ] && continue
  log "speed $f"
  ( [ $f = lm ] && export SPEED_FALLBACK=lm; HARNESS_NORM=$NORM SPEED_GPU_ONLY=1 SPEED_NAME=${L}_idle_$f \
    taskset -c $C $PY $H/bench/speed.py $RUN 32 ) || log "FAILED speed $f"
done
if [ ! -e $OUT/cyc1000_$L.json ]; then
  log "1000-day run: start"
  HARNESS_NORM=$NORM taskset -c $C $PY $H/bench/cyc1000.py $RUN $OUT/cyc1000_$L.json 32; log "1000-day run: exit $?"
fi
bash $H/run/bench_dev.sh $RUN $L $C "PD PX P2l P2s PH DEV6 T2 SD0.1 SD1"
log "CANDIDATE-BENCHED"
CUDA_VISIBLE_DEVICES= SCORE_RULE=phys HARNESS_NORM= taskset -c $C $PY $H/bench/judge.py $L $REF
log "CANDIDATE-DONE"
