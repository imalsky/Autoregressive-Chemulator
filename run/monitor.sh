#!/bin/bash
# Sprint monitor: exits (printing why) at the first new event: a failed lane job, a FATAL line in a CPU log, a NaN in a
# training log, or a milestone marker not yet seen (listed in monitor.seen). Re-run after handling the event.
# 2026-10-09: X (sprint _build) and R (runs) from env.sh; the watched files stay the sprint's.
source /home/imalsky/Chemulator_Project/code/e13-harness/run/env.sh
SEEN=$X/monitor.seen; touch $SEEN
while true; do
  for f in $X/q/*/failed/*.sh; do [ -e "$f" ] && ! grep -qxF "$f" $SEEN && { echo "$f" >> $SEEN; echo "EVENT failed job $f"; exit 0; }; done
  for f in $X/run_cpu2.log $X/run_cpu3.log $X/gen_d/drive*.log $X/gen_h/drive_h*.log $X/gen_shift/run_m1.log $X/gen_shift/build_m_cache.log $X/truth/run_truth.log $X/paths_v2/drive.log $X/gen_md/build_md_cache.log; do
    [ -e $f ] && grep -q FATAL $f && ! grep -qxF "FATAL $f" $SEEN && { echo "FATAL $f" >> $SEEN; echo "EVENT FATAL in $f"; exit 0; }
  done
  for f in $R/e13_*/log.csv; do
    [ $(find $f -mmin -30 2>/dev/null | wc -l) = 1 ] && tail -3 $f | grep -qi nan && ! grep -qxF "NAN $f" $SEEN && { echo "NAN $f" >> $SEEN; echo "EVENT NaN in $f"; exit 0; }
  done
  for m in "M-CACHE-DONE:$X/gen_shift/build_m_cache.log" "V-:$X/run_cpu3.log" "D-CACHE-DONE:$X/run_cpu3.log" "PATHS-V2-DONE:$X/paths_v2/drive.log" \
           "OK 14_M_g500:$X/q/bench/runner.log" "OK 15_M_paths:$X/q/bench/runner.log" "OK 13_S_g500:$X/q/bench/runner.log" \
           "OK 12_D1a_g500:$X/q/bench/runner.log" "OK 11_C1_g500:$X/q/bench/runner.log" "OK 16_T_g500:$X/q/bench/runner.log" \
           "OK 18_D1b_g500_s1:$X/q/bench/runner.log" "OK 12a_G500:$X/q/bench/runner.log" "OK 12b_G500:$X/q/bench/runner.log" \
           "OK 12c_G500:$X/q/bench/runner.log" "OK 12d_G500:$X/q/bench/runner.log" "OK 90_e13_big17lr_s0:$X/q/laneB/runner.log" \
           "OK 07l_P2l:$X/q/bench/runner.log" "OK 08_P2s:$X/q/bench/runner.log" "OK 091_P2s:$X/q/bench/runner.log" "OK 095_P2s:$X/q/bench/runner.log" "OK 110_G500:$X/q/bench/runner.log" "OK 111_G500:$X/q/bench/runner.log" \
           "OK 112_G500:$X/q/bench/runner.log" "OK 113_G500:$X/q/bench/runner.log" "OK 096l_P2l:$X/q/bench/runner.log" "M64-CACHE-DONE:$X/gen_shift/build_m64_cache.log" "MD-CACHE-DONE:$X/gen_md/build_md_cache.log" "OK 114_G500:$X/q/bench/runner.log" "OK 115_G500:$X/q/bench/runner.log" "OK 116_G500:$X/q/bench/runner.log" "OK 117_G500:$X/q/bench/runner.log" "OK 118_G500:$X/q/bench/runner.log" "OK 3z4_Gc:$X/q/bench/runner.log" "OK 3z5_Gc:$X/q/bench/runner.log" "OK 3z6_Gc:$X/q/bench/runner.log" "OK 3z7_Gc:$X/q/bench/runner.log" "OK 3z8_Gc:$X/q/bench/runner.log" "OK 11a_G500:$X/q/bench/runner.log" "OK 119_G500:$X/q/bench/runner.log" "OK 3z9_Gc:$X/q/bench/runner.log" "OK 3za_Gc:$X/q/bench/runner.log" "OK 11b_G500:$X/q/bench/runner.log" "OK 11c_G500:$X/q/bench/runner.log" "OK 3zb_Gc:$X/q/bench/runner.log" "OK 3zc_Gc:$X/q/bench/runner.log" "OK 67_e13_f0fs_s0:$X/q/laneA/runner.log" "OK 68_e13_f0fs_s1:$X/q/laneB/runner.log"; do
    k=${m%%:*}; f=${m#*:}
    [ "$k" = "V-" ] && k="V: VERDICT"
    [ -e $f ] && grep -q "$k" $f && ! grep -qxF "MILESTONE $k" $SEEN && { echo "MILESTONE $k" >> $SEEN; echo "EVENT milestone: $k"; exit 0; }
  done
  # 2026-10-09: generic idle event (the per-job milestones missed the branch, R-full and xlong jobs, and the lanes sat
  # idle ~13 h): every lane with no pending and no running job, keyed by the total done count so it fires once per state
  n=$(ls $X/q/*/pending/*.sh $X/q/*/running/*.sh 2>/dev/null | wc -l); k="IDLE $(ls $X/q/*/done/*.sh 2>/dev/null | wc -l)"
  [ "$n" = 0 ] && ! grep -qxF "$k" $SEEN && { echo "$k" >> $SEEN; echo "EVENT all lanes idle ($k)"; exit 0; }
  f=$X/paths_v2/rtolchk.log; [ -e $f ] && grep -q RTOLCHK-DONE $f && ! grep -qxF "MILESTONE RTOLCHK" $SEEN && { echo "MILESTONE RTOLCHK" >> $SEEN; echo "EVENT milestone: RTOLCHK-DONE"; exit 0; }
  sleep 120
done
