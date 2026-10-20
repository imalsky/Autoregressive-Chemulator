#!/bin/bash
# Development benches of one model (scale phase q/bench6.sh, plus the T2 and SD sets of the candidate drivers p/cbg.sh and
# p/cbgp.sh, same flags). Every set runs this repository's lib/lhbench.py with the P* inference policy (--modes P3
# --closed-iters 32 --closed-fallback lm --lm-adaptive), ARCH_CACHE and HARNESS_NORM = arch_gcm17m_res; path sets with
# LHB_BLOCK 50, the combined and T2 / SD arm-H benches with LHB_BLOCK ${BLK_H:-10} (memory). Never change the rows per call
# of a bench: results differ by up to 7e-4 dex across batch shapes. A bench whose meta.json exists is skipped.
#   PD PX P2l P2s PH   path sets: DEV (paths_v3/dev), xlong (paths_v2x), v2l, v2s, the opened path holdout (paths_v3/hold)
#   DEV6               the six fixed-condition sets G + Gc + H2 + H2c + H3 + H3c as ONE combined H bench on the devall6 tag
#                      (datagen/combine_tags.py), split back into the per-set files by bench/split6.py
#   T2                 the snap-back tag (paths.T2/tag), then bench/score_t2.py -> paths.T2/score_t2_<label>.{json,txt}
#   SD0.1 SD1          10,000 calls of dt 0.1 / 1 s from VULCAN's states at 1e3 and 1e6 s on devall6 (bench/sd_score.py)
#   PH2 H4ALL          the sealed path set and fixed-condition sets: opened once, on a frozen recipe, by the author's decision
#   bash bench_dev.sh <run dir> <label> <CPUs> [sets, default "PD PX P2l P2s PH DEV6"]   (GPU1; logs in $LOGS)
source "$(dirname "$(readlink -f "$0")")/env.sh"
set -u
P=$CHEM_PROJECT/experiments/2026-10-09_e13-conservation/_build/paths_v3
T2=$CHEM_PROJECT/experiments/2026-10-13_e13-scale/_build/t2
DEVALL6=$CHEM_SCRATCH/e13_scaledev/devall6
LOGS=${LOGS:-$CHEM_PROJECT/experiments/2026-10-13_e13-scale/_build/bench/logs}   # paths.OUT/logs
NORM=$CHEM_SCRATCH/arch_gcm17m_res/norm.json
M=$1 L=$2 A=$3 SETS=${4:-"PD PX P2l P2s PH DEV6"}
mkdir -p $LOGS
lb() {  # lb <block> <out> <args...>
  local blk=$1 out=$2; shift 2
  [ -e $out/meta.json ] && return 0
  LHB_BLOCK=$blk ARCH_CACHE=$CHEM_SCRATCH/arch_gcm17m_res HARNESS_NORM=$NORM taskset -c $A $PY $H/lib/lhbench.py $M $out SP \
    --ckpt stage2.pt --modes P3 --closed-iters 32 --closed-fallback lm --lm-adaptive "$@" > $LOGS/$(basename $out).log 2>&1 \
    || { echo "$(date '+%F %T') FAILED $(basename $out)"; return 1; }
  echo "$(date '+%F %T') done $(basename $out): $(grep -h 'projection failures' $LOGS/$(basename $out).log | tail -1)"
}
hs() {  # hs <tag dir> : combined H bench, then split
  local out=$E/H_$(basename $1)_$L
  lb ${BLK_H:-10} $out --arms H --h-tag $1 --h-starts 1e-3,1e3,1e6 --h-dt 500 --startup-substeps 5:1 || return 1
  $PY $H/bench/split6.py $out $L || { echo "$(date '+%F %T') FAILED split $(basename $out)"; return 1; }
  echo "$(date '+%F %T') split $(basename $out)"
}
cd $H/lib
for s in $SETS; do
  case $s in
    PD)  lb 50 $E/PD_$L --arms P --p-dir $P/dev ;;
    PX)  lb 50 $E/PX_$L --arms P --p-dir $E/paths_v2x ;;
    P2l) lb 50 $E/P2l_$L --arms P --p-dir $E/paths_v2l ;;
    P2s) lb 50 $E/P2s_$L --arms P --p-dir $E/paths_v2s ;;
    PH)  lb 50 $E/PH_$L --arms P --p-dir $P/hold ;;
    PH2) lb 50 $E/PH2_$L --arms P --p-dir $P/hold2 ;;
    DEV6) [ -e $E/H_g500_$L/meta.json ] || hs $DEVALL6 ;;
    H4ALL) hs $CHEM_SCRATCH/e13_holdout4/h4all ;;
    T2) lb ${BLK_H:-10} $E/H_t2tag_$L --arms H --h-tag $T2/tag --h-starts 0 --h-dt 500 --h-calls 172800 --h-times $T2/times_t2.npy \
          && { [ -e $T2/score_t2_$L.json ] || CUDA_VISIBLE_DEVICES= SCORE_RULE=phys HARNESS_NORM= taskset -c $A $PY $H/bench/score_t2.py \
          $E/H_t2tag_$L/H_tag_dt500.npz $T2/score_t2_$L > $LOGS/score_t2_$L.log 2>&1 || echo "$(date '+%F %T') FAILED score_t2 $L"; } ;;
    SD0.1|SD1) dt=${s#SD}; lb ${BLK_H:-10} $E/H_sd_dt${dt}_$L --arms H --h-tag $DEVALL6 --h-starts 1e3,1e6 --h-dt $dt --h-calls 10000 ;;
    *) echo "unknown set $s"; exit 1 ;;
  esac
done
echo "$(date '+%F %T') ALL-DONE $L"
