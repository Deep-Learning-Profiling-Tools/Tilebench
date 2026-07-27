#!/bin/bash
# Targeted section set at a reduced problem size.
#
# `--set full` with PM sampling, and even the wider section set at
# M=N=4096 K=8192, need too many replay passes: each pass re-saves ~160 MB of
# input/output buffers and the profile ran >10 min per kernel. M=N=2048 K=4096
# keeps every backend in the same steady-state regime (K-loop still 64 tiles
# deep at BK=64) while profiling in seconds.
set -e
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
mkdir -p reports logs

SECTIONS="--section SpeedOfLight --section ComputeWorkloadAnalysis \
--section WarpStateStats --section Occupancy --section LaunchStats"

M="${M:-2048}"
N="${N:-2048}"
K="${K:-4096}"
DTYPES="${DTYPES:-fp16 fp8}"
for dt in $DTYPES; do
  for backend in tilelang tilelang_ws triton cutile; do
    name="${backend}_${dt}"
    echo "=== $name (M=$M N=$N K=$K) ==="
    $NCU --profile-from-start off $SECTIONS \
         --target-processes all --force-overwrite \
         -o "reports/${name}" \
         $PY harness/profile_matmul.py --backend "$backend" --dtype "$dt" \
            --M "$M" --N "$N" --K "$K" \
         > "logs/${name}.log" 2>&1
  done
done
echo done
