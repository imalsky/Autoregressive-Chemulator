#!/bin/bash
# How arch_gcm17m_res was built: the 53-species and 25-entry
# reservoir caches aligned with the frozen model's 12-species cache arch_gcm17m_all (b00 + b01 + m18). arch_gcm17m_res's
# norm.json is CBg's normalization (irreplaceable): this script refuses to run while that cache exists. The 12-species
# source cache and the 53-species intermediate are not kept; rebuild arch_gcm17m_all with cache_gcm.py first.
#   bash build_res17m.sh <CPU>
source "$(dirname "$(dirname "$(readlink -f "$0")")")/run/env.sh"
set -eu
C=${1:-82}
D=$CHEM_DATA/caches
[ -e $D/arch_gcm17m_res/meta.json ] && { echo "refusing: $D/arch_gcm17m_res exists"; exit 1; }
date; taskset -c $C $PY $H/datagen/cache_align53.py $D/arch_gcm17m_all $D/arch_gcm17m_all53 $CHEM_DATA/raw/raw_gcm_m
date; taskset -c $C $PY $H/datagen/cache_res.py $D/arch_gcm17m_all53 $D/arch_gcm17m_res
date; echo RES17M-DONE
