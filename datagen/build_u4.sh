#!/bin/bash
# build_u4.sh (2026-10-17): the training cache U4 = U3's 14 files (links, unchanged order) + run_gcm_z23n1 (78,106 runs) +
# run_gcm_z23n2 (69,836 listed rows) from today's targeted data (gen/rows_n1.py, rows_n2.py: Z 1-100x, C/O 0.3-1.0, 1e-5 to
# 100 bar). Same code and steps as t5/build_u3.sh: cache_gcm.py chunks (odd CPUs 41-55) + merge with
# GCM_KINDS=eq,mix,pert,broad,kick -> caches/arch_gcmU4_all; u4/check_u4.py (prefix identity vs U3_all, per-file
# accounting, n1 / n2 coverage); cache_align53.py -> caches/arch_gcmU4_all53 (/home); scratch quota >= 17 GiB; cache_res.py
# -> /scratch/imalsky/arch_gcmU4_res; t_grid check; check_u4.py --res vs U3_res; size < 18 GB. Rerunnable (finished steps
# and U4-CACHE-DONE are skipped); FATAL on failure. The U3 caches the checks compare with were deleted on 2026-10-19, so a
# rebuild from scratch stops at check_u4 until they are rebuilt. cache_gcm.py now takes each group's start type from its
# 'start' attr, so the generator's extended gcm.py is no longer needed (its check is dropped).
#   setsid nohup bash build_u4.sh > build_u4.log 2>&1 &
set -u
source "$(dirname "$(dirname "$(readlink -f "$0")")")/run/env.sh"
export CUDA_VISIBLE_DEVICES=""
N=$CHEM_PROJECT/experiments/2026-10-13_e13-scale/_build
U=$N/u4
DA=$H/data
RD=$DA/raw_gcm_n/U4
C12=$N/caches/arch_gcmU4_all; C53=$N/caches/arch_gcmU4_all53; CR=$CHEM_SCRATCH/arch_gcmU4_res
f() { echo "$(date '+%F %T') FATAL build_u4: $*"; exit 1; }
[ -e $U/build/U4-CACHE-DONE ] && { echo "$(date '+%F %T') U4-CACHE-DONE exists"; exit 0; }
mkdir -p $RD || f "mkdir $RD"
for l in $(ls $DA/raw_gcm_k/U3); do
  t=$(readlink $DA/raw_gcm_k/U3/$l); [ -f $t ] || f "U3 link target missing: $t"
  [ -L $RD/$l ] || ln -s $t $RD/$l || f "link $l"
  [ "$(readlink $RD/$l)" = "$t" ] || f "link $l points elsewhere"
done
for n in n1 n2; do
  t=$DA/raw_gcm_n/run_gcm_z23$n-result.h5; [ -f $t ] && [ ! -L $t ] || f "not a regular file: $t"
  [ -L $RD/run_gcm_z23$n-result.h5 ] || ln -s $t $RD/run_gcm_z23$n-result.h5 || f "link z23$n"
done
F=$(ls $RD | sort | paste -sd,); K=$(ls $RD | wc -l)
[ $K -eq 16 ] || f "expected 16 files, got $K: $F"
echo "$(date '+%F %T') U4 files ($K): $F"
if [ ! -e $C12/meta.json ]; then
  CH=(41 43 45 47 49 51 53 55); pids=()
  for i in $(seq 0 $((2 * K - 1))); do
    GCM_RAW=$RD GCM_FILES=$F GCM_KINDS=eq,mix,pert,broad,kick RAWTRAJ_FLOOR=1 ARCH_CACHE=$C12 \
      taskset -c ${CH[$((i % 8))]} $PY $H/datagen/cache_gcm.py chunk $K 2 $i &
    pids+=($!)
    if [ ${#pids[@]} -eq 8 ]; then for p in "${pids[@]}"; do wait $p || f "chunk"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait $p || f "chunk"; done
  GCM_RAW=$RD GCM_FILES=$F GCM_KINDS=eq,mix,pert,broad,kick RAWTRAJ_FLOOR=1 ARCH_CACHE=$C12 taskset -c 57 $PY \
    $H/datagen/cache_gcm.py merge $K 2 || f "merge"
fi
echo "$(date '+%F %T') 12-species cache done"
mkdir -p $U/build
if ! grep -q "CHECK-U4 OK" $U/build/check_u4.txt 2>/dev/null; then
  taskset -c 57 $PY $H/datagen/check_u4.py $C12 $N/caches/arch_gcmU3_all $RD > $U/build/check_u4.txt 2>&1
  grep -q "CHECK-U4 OK" $U/build/check_u4.txt || f "check_u4 (see u4/build/check_u4.txt)"
fi
echo "$(date '+%F %T') CHECK-U4 OK"
[ -e $C53/meta.json ] || taskset -c 57 $PY $H/datagen/cache_align53.py $C12 $C53 $RD || f "align53"
echo "$(date '+%F %T') 53-species cache done"
if [ ! -e $CR/meta.json ]; then
  read used limit <<< $(quota -w 2>/dev/null | awk '$1 == "/dev/nvme0n1p1" {gsub(/\*/, "", $2); print $2, $4}')
  [ -n "${used:-}" ] && [ -n "${limit:-}" ] || f "cannot read the /scratch quota"
  free_kb=$((limit - used)); echo "$(date '+%F %T') /scratch quota: used $used KiB of $limit KiB, free $free_kb KiB"
  [ $free_kb -ge $((17 * 1024 * 1024)) ] || f "/scratch free $free_kb KiB < 17 GiB"
  taskset -c 57 $PY $H/datagen/cache_res.py $C53 $CR || f "res"
fi
cmp $CR/t_grid.npy $CHEM_SCRATCH/arch_gcm17m_res/t_grid.npy || f "t_grid differs"
SZ=$(du -sb $CR | cut -f1); echo "$(date '+%F %T') U4 res cache $SZ bytes"
[ $SZ -lt 18000000000 ] || f "U4 res cache $SZ bytes >= 18 GB"
if ! grep -q "CHECK-U4-RES OK" $U/build/check_u4_res.txt 2>/dev/null; then
  taskset -c 57 $PY $H/datagen/check_u4.py --res $CR $CHEM_SCRATCH/arch_gcmU3_res > $U/build/check_u4_res.txt 2>&1
  grep -q "CHECK-U4-RES OK" $U/build/check_u4_res.txt || f "check_u4 --res (see u4/build/check_u4_res.txt)"
fi
echo "$(date '+%F %T') CHECK-U4-RES OK"
echo "U4-CACHE-DONE $(date '+%F %T')" > $U/build/U4-CACHE-DONE
echo "$(date '+%F %T') U4-CACHE-DONE"
