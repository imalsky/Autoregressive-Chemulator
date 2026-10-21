#!/bin/bash
# The clean training cache from the clean.py packs (build_u4.sh's steps without its comparisons with an older cache):
#   1. cache_gcm.py: GCM_K x GCM_NPER chunks on the CLEAN_CPUS (as many at a time as CPUs listed) + merge, with
#      GCM_KINDS=eqt,eqn,eqf,mix,pert,rawA,rawB -> $CLEAN_WORK/all12 (12 species);
#   2. check_clean.py: one 'file' check per pack (in parallel on the CLEAN_CPUS) and the 'cache' check;
#   3. cache_align53.py -> $CLEAN_WORK/all53 (53 species); then all12 is deleted;
#   4. free quota on CLEAN_RES's filesystem >= 17 GiB; cache_res.py -> $CLEAN_RES (25 entries + norm.json); its t_grid must
#      equal arch_gcm17m_res's byte for byte and its size be < 18 GB; then all53 is deleted.
# A deletion removes exactly the five files a cache step writes (raw_<split>.npz, t_grid.npy, meta.json) and then the
# folder, after checking that the folder holds nothing else; the list with sizes goes to $CLEAN_WORK/delete_<name>.txt.
# CLEAN_KEEP=1 keeps all12 and all53. Rerunnable: each finished step leaves a marker in $CLEAN_WORK/markers and is
# skipped; CLEAN_RES is refused if it exists and this work folder did not start it. CLEAN-CACHE-DONE ends a build;
# FATAL on any failure.
# Environment (defaults = the full clean set): GCM_RAW ($CHEM_DATA/raw/raw_clean), GCM_FILES (run_clean_b00-result.h5 ...
# b11), GCM_K (12), GCM_NPER (2; cache_gcm.py asserts GCM_K x GCM_NPER chunks), CLEAN_RES ($CHEM_DATA/caches/arch_clean_res),
# CLEAN_DESIGN_MAP ($CHEM_DATA/gen/clean/design_map.npy), CLEAN_CPUS (54,...,61).
# CLEAN_WORK (no default) holds all12, all53, the check outputs, markers and deletion lists.
#   CLEAN_WORK=<dir> setsid nohup bash build_clean.sh > <dir>/build_clean.log 2>&1 < /dev/null &
set -u
source "$(dirname "$(dirname "$(readlink -f "$0")")")/run/env.sh"
export CUDA_VISIBLE_DEVICES=""
f() { echo "$(date '+%F %T') FATAL build_clean: $*"; exit 1; }
say() { echo "$(date '+%F %T') $*"; }
[ -n "${CLEAN_WORK:-}" ] || f "CLEAN_WORK is not set"
W=$CLEAN_WORK
RAW=${GCM_RAW:-$CHEM_DATA/raw/raw_clean}
FILES=${GCM_FILES:-$(for b in $(seq -w 0 11); do echo run_clean_b$b-result.h5; done | paste -sd,)}
K=${GCM_K:-12}; NPER=${GCM_NPER:-2}
CR=${CLEAN_RES:-$CHEM_DATA/caches/arch_clean_res}
DMAP=${CLEAN_DESIGN_MAP:-$CHEM_DATA/gen/clean/design_map.npy}
IFS=, read -r -a CPUS <<< "${CLEAN_CPUS:-54,55,56,57,58,59,60,61}"
IFS=, read -r -a FL <<< "$FILES"
KINDS=eqt,eqn,eqf,mix,pert,rawA,rawB
C12=$W/all12; C53=$W/all53; MK=$W/markers
for p in "$W" "$RAW" "$CR" "$DMAP"; do
  case "$p" in *..*|*SEALED*|*e13_holdout4*|*hold2*) f "refused path: $p" ;; esac
done
[ -e $MK/CLEAN-CACHE-DONE ] && { say "CLEAN-CACHE-DONE exists ($MK)"; exit 0; }
mkdir -p $MK $W/check || f "mkdir $W"
[ ${#FL[@]} -eq $K ] || f "GCM_K $K but ${#FL[@]} files: $FILES"
for x in "${FL[@]}"; do [ -f $RAW/$x ] && [ ! -L $RAW/$x ] || f "not a regular file: $RAW/$x"; done
[ -f $DMAP ] || f "design table missing: $DMAP"
say "files ($K, $NPER chunks each, CPUs ${CPUS[*]}): $FILES"

delcache() {  # delcache <cache dir> <name>
  local d=$1 m=$W/delete_$2.txt fs="meta.json raw_test.npz raw_train.npz raw_validation.npz t_grid.npy"
  if [ ! -e $d ] && [ -s $m ]; then say "$d already deleted (list $m)"; return 0; fi
  case "$d" in /home/imalsky/*) ;; *) f "delete $d: outside /home/imalsky" ;; esac
  [ "$(ls -A $d | LC_ALL=C sort | paste -sd' ')" = "$fs" ] || f "delete $d: the folder holds other files: $(ls -A $d | paste -sd' ')"
  : > $m || f "write $m"
  for x in $fs; do
    [ -f $d/$x ] && [ ! -L $d/$x ] || f "delete $d/$x: not a regular file"
    echo "$(stat -c %s $d/$x) $d/$x" >> $m
  done
  say "deleting $d (list $m):"; cat $m
  for x in $fs; do rm $d/$x || f "rm $d/$x"; done
  rmdir $d || f "rmdir $d"
}

if [ ! -e $MK/c12.done ]; then
  pids=()
  for i in $(seq 0 $((K * NPER - 1))); do
    GCM_RAW=$RAW GCM_FILES=$FILES GCM_KINDS=$KINDS RAWTRAJ_FLOOR=1 ARCH_CACHE=$C12 \
      taskset -c ${CPUS[$((i % ${#CPUS[@]}))]} $PY $H/datagen/cache_gcm.py chunk $K $NPER $i &
    pids+=($!)
    if [ ${#pids[@]} -eq ${#CPUS[@]} ]; then for p in "${pids[@]}"; do wait $p || f "chunk"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait $p || f "chunk"; done
  GCM_RAW=$RAW GCM_FILES=$FILES GCM_KINDS=$KINDS RAWTRAJ_FLOOR=1 ARCH_CACHE=$C12 taskset -c ${CPUS[0]} $PY \
    $H/datagen/cache_gcm.py merge $K $NPER || f "merge"
  touch $MK/c12.done
fi
say "12-species cache done"
if [ ! -e $MK/check.done ]; then
  pids=(); n=0
  for x in "${FL[@]}"; do
    taskset -c ${CPUS[$((n % ${#CPUS[@]}))]} $PY $H/datagen/check_clean.py file $C12 $RAW $DMAP $x > $W/check/$x.txt 2>&1 &
    pids+=($!); n=$((n + 1))
    if [ ${#pids[@]} -eq ${#CPUS[@]} ]; then for p in "${pids[@]}"; do wait $p; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait $p; done
  for x in "${FL[@]}"; do grep -q "^CHECK-CLEAN-FILE OK $x$" $W/check/$x.txt || f "check_clean file $x (see $W/check/$x.txt)"; done
  GCM_FILES=$FILES taskset -c ${CPUS[0]} $PY $H/datagen/check_clean.py cache $C12 > $W/check/cache.txt 2>&1
  grep -q "^CHECK-CLEAN-CACHE OK$" $W/check/cache.txt || f "check_clean cache (see $W/check/cache.txt)"
  touch $MK/check.done
fi
say "CHECK-CLEAN OK"
if [ ! -e $MK/c53.done ]; then
  taskset -c ${CPUS[0]} $PY $H/datagen/cache_align53.py $C12 $C53 $RAW || f "align53"
  touch $MK/c53.done
fi
say "53-species cache done"
if [ ! -e $MK/del12.done ]; then
  if [ "${CLEAN_KEEP:-}" = 1 ]; then say "CLEAN_KEEP=1: $C12 kept"; else delcache $C12 all12; fi
  touch $MK/del12.done
fi
if [ ! -e $MK/res.done ]; then
  [ ! -e $CR ] || [ -e $MK/res.started ] || f "$CR exists and was not started by this work folder; refusing to write into it"
  par=$(dirname $CR); dev=$(df --output=source $par | tail -1)
  read used limit <<< $(quota -w 2>/dev/null | awk -v d=$dev '$1 == d {gsub(/\*/, "", $2); print $2, $4}')
  [ -n "${used:-}" ] && [ -n "${limit:-}" ] || f "cannot read the quota of $dev ($par)"
  free_kb=$((limit - used)); say "quota of $dev: used $used KiB of $limit KiB, free $free_kb KiB"
  [ $free_kb -ge $((17 * 1024 * 1024)) ] || f "free $free_kb KiB < 17 GiB on $dev"
  touch $MK/res.started
  taskset -c ${CPUS[0]} $PY $H/datagen/cache_res.py $C53 $CR || f "res"
  cmp $CR/t_grid.npy $CHEM_DATA/caches/arch_gcm17m_res/t_grid.npy || f "t_grid differs from arch_gcm17m_res's"
  SZ=$(du -sb $CR | cut -f1); say "res cache $CR: $SZ bytes"
  [ $SZ -lt 18000000000 ] || f "res cache $SZ bytes >= 18 GB"
  touch $MK/res.done
fi
say "res cache done: $CR"
if [ ! -e $MK/del53.done ]; then
  if [ "${CLEAN_KEEP:-}" = 1 ]; then say "CLEAN_KEEP=1: $C53 kept"; else delcache $C53 all53; fi
  touch $MK/del53.done
fi
echo "CLEAN-CACHE-DONE $(date '+%F %T') $CR" > $MK/CLEAN-CACHE-DONE
say "CLEAN-CACHE-DONE"
