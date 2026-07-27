#!/bin/bash
# Scheduler-level capture: Active / Eligible / Issued warps per scheduler.
#
# The matched run (run_ncu_matched.sh) omitted SchedulerStats, so
# smsp__warps_eligible was never collected. Without it the absolute stall
# breakdown cannot distinguish "warps are stalled" from "the scheduler had
# nothing eligible to run" -- and TileLang carries 2.83 active warps/scheduler
# against Triton's 0.99, so raw warp-cycle totals are not comparable.
#
# Same shape and matched tiles as run_ncu_matched.sh. SpeedOfLight and
# ComputeWorkloadAnalysis are dropped since m_*.ncu-rep already has them; this
# keeps the replay-pass count from growing.
set -e
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
mkdir -p reports logs

SECTIONS="--section SchedulerStats --section WarpStateStats \
--section Occupancy --section LaunchStats"

M="${M:-4096}"
N="${N:-4096}"
K="${K:-2048}"
DTYPES="${DTYPES:-fp16 fp8}"
for dt in $DTYPES; do
  for backend in tilelang_ws triton cutile; do
    name="sch_${backend}_${dt}"
    echo ">>> $name (M=$M N=$N K=$K, matched tile)"
    $NCU --profile-from-start off $SECTIONS \
         --target-processes all --force-overwrite \
         -o "reports/${name}" \
         $PY -u harness/profile_matmul.py --backend "$backend" --dtype "$dt" \
            --M "$M" --N "$N" --K "$K" --matched \
      2>&1 | tee "logs/${name}.log" | grep --line-buffered -E "max_abs_err|rror|==PROF== Disc" || true
    echo "<<< $name done"
  done
done
echo ALLDONE
