#!/bin/bash
source "$(dirname "$0")/env.sh"; cd "$RUNDIR"
SECT="--section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats \
--section Occupancy --section LaunchStats --section SchedulerStats \
--section MemoryWorkloadAnalysis --section InstructionStats"

cap () { # tag backend dtype block extra...
  local tag=$1 b=$2 dt=$3 blk=$4; shift 4
  echo ">>> $tag"
  $NCU --profile-from-start off $SECT --target-processes all --force-overwrite \
       -o "reports/${tag}" \
       $PY -u harness/profile_sigmoid.py --backend "$b" --dtype "$dt" --block "$blk" "$@" \
    2>&1 | tee "logs/${tag}.log" | grep -E "max_abs_err|==ERROR" || true
  echo "<<< $tag"
}

# 1. tuned winners, fp16 (the configuration that produces the observed 1.46x)
cap tuned_tilelang_fp16 tilelang fp16 1024 --threads 128
cap tuned_triton_fp16   triton   fp16 4096 --warps 8
cap tuned_cutile_fp16   cutile   fp16 4096 --occupancy 16

# 2. tuned winners, fp32 -- all three tuners picked 2048, so already matched.
#    This is the control: TileLang WINS here.
cap tuned_tilelang_fp32 tilelang fp32 2048 --threads 256
cap tuned_triton_fp32   triton   fp32 2048 --warps 8
cap tuned_cutile_fp32   cutile   fp32 2048 --occupancy 8

# 3. matched BLOCK_SIZE=2048 at fp16 -- controls the search-space cap confound
cap match_tilelang_fp16 tilelang fp16 2048 --threads 256
cap match_triton_fp16   triton   fp16 2048 --warps 8
cap match_cutile_fp16   cutile   fp16 2048 --occupancy 8
echo ALLDONE
