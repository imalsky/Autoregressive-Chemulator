#!/bin/bash
# Lane runner: runs the jobs in <lane>/pending/*.sh one at a time, in name order, picking the first whose '# needs:'
# paths all exist; each job gets CPUS=<cpus>. Waits while another user holds a compute process on GPU1 (B5:00.0).
# Stops when <lane>/STOP exists (after the running job). Jobs move to running/, then done/ or failed/ with their log.
#   bash runner.sh <lane_dir> <cpus>
L=$1; C=$2
source /home/imalsky/Chemulator_Project/code/e13-harness/run/env.sh
mkdir -p $L/pending $L/running $L/done $L/failed
gpu_free() {
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' '$2 ~ /B5:00.0/ {print $1}'); do
    u=$(ps -o user= -p $p 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" != imalsky ] && return 1
  done; return 0; }
echo "$(date) runner $L on CPUs $C"
while true; do
  [ -e $L/STOP ] && { echo "$(date) STOP"; break; }
  job=""
  for f in $(ls $L/pending/*.sh 2>/dev/null | sort); do
    ok=1; for n in $(sed -n 's/^# needs: //p' $f); do [ -e "$n" ] || ok=0; done
    [ $ok = 1 ] && { job=$f; break; }
  done
  if [ -z "$job" ] || ! gpu_free; then sleep 60; continue; fi
  b=$(basename $job .sh); mv $job $L/running/
  echo "$(date) START $b"
  if CPUS=$C bash $L/running/$b.sh > $L/running/$b.log 2>&1; then mv $L/running/$b.sh $L/running/$b.log $L/done/; echo "$(date) OK $b"
  else mv $L/running/$b.sh $L/running/$b.log $L/failed/; echo "$(date) FAILED $b"; fi
done
