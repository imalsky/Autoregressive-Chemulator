#!/bin/bash
# Step-size sweep driver (2026-10-19; author: "no dt gap"): one model at a time on GPU1. Rows = paths.DTSWEEP/rows_main.json
# (600 dev jobs + the 6 fig4 cases, bench/sweep/select_rows.py) x dt 1,2,5,10,20,50,100,200,500 s x starts 1e-3, 1e6 s =
# 10,908 rows, N = 10,000 calls (about 5.4 min per model at 32 ms per call), common elapsed times 1e3 and 1e4 s recorded.
# bench/sweep/sweep.py runs this repository's lib/lhbench.py (the stored sweeps of 2026-10-19 ran each run's frozen copy,
# <run>/harness/lib: pass --lib for that). Outputs paths.DTSWEEP/sw_<run>/, logs paths.DTSWEEP/logs/; a model whose output
# exists is skipped. Score with bench/sweep/score_sweep.py.
#   CPUS=25,27,29,31 setsid nohup bash run_sweep.sh [run name ...] > <log> 2>&1 &
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$CHEM_PROJECT/experiments/2026-10-13_e13-scale/_build/dtsweep
B=$CHEM_PROJECT/experiments/2026-10-13_e13-scale/_build/runs
RUNS=${*:-e13s_CBg_s2 e13s_LD1_s3 e13s_FINw1536_s2 e13s_LD_s3}
n=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$n" = 0 ] || { echo "GPU1 busy ($n processes); stop"; exit 1; }
mkdir -p $S/logs
for r in $RUNS; do
  [ -e $S/sw_$r/sweep.npz ] && { echo "skip $r (done)"; continue; }
  echo "$(date '+%F %T') start $r"
  taskset -c ${CPUS:-25,27,29,31} $PY $H/bench/sweep/sweep.py $B/$r $S/sw_$r $S/rows_main.json --n 10000 > $S/logs/sw_$r.log 2>&1
  echo "$(date '+%F %T') exit $? $r: $(tail -1 $S/logs/sw_$r.log)"
done
echo SWEEP-DONE
