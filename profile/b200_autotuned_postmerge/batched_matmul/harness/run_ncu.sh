#!/bin/bash
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
SECTIONS="--section SpeedOfLight --section ComputeWorkloadAnalysis \
--section WarpStateStats --section Occupancy --section LaunchStats \
--section SchedulerStats"
M="${M:-640}"; DT="${DT:-fp16}"
for b in tilelang triton cutile torch; do
  n="c_${b}_${DT}_M${M}"
  echo ">>> $n"
  $NCU --profile-from-start off $SECTIONS --target-processes all --force-overwrite \
       -o "reports/${n}" \
       $PY -u harness/profile_bmm.py --backend "$b" --dtype "$DT" --M "$M" \
    2>&1 | tee "logs/${n}.log" | grep -E "max_abs_err|==PROF== Disc|rror" || true
  echo "<<< $n done"
done
echo ALLDONE
