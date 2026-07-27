#!/bin/bash
source "$(dirname "$0")/env.sh"; cd "$RUNDIR"
SECT="--section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats \
--section Occupancy --section LaunchStats --section SchedulerStats --section MemoryWorkloadAnalysis"
for b in triton tilelang cutile; do
  echo ">>> $b"
  $NCU --profile-from-start off $SECT --target-processes all --force-overwrite \
       -o "reports/${b}_fp16" \
       $PY -u harness/profile_wd.py --backend "$b" --dtype fp16 --M 8192 \
    2>&1 | tee "logs/${b}.log" | grep -E "max_abs_err|Disc" || true
  echo "<<< $b"
done
echo ALLDONE
