#!/bin/bash
# Same as run_ncu.sh but every backend is forced to the recorded tuned tile
# (fp16 256x256x64, fp8 256x256x128), so the comparison isolates codegen.
#
# M=N=4096 rather than 2048: at a 256x256 tile, 2048 gives only 64 CTAs on 148
# SMs, i.e. a single partial wave, which distorts occupancy and SOL. 4096 gives
# 256 CTAs. K=2048 keeps every replay pass's buffer save under ~64 MB;
# adding MemoryWorkloadAnalysis, InstructionStats, or K=4096 pushes the pass
# count into a hang. Opcode histograms come from run_ncu_opcodes.sh instead.
set -e
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
mkdir -p reports logs

SECTIONS="--section SpeedOfLight --section ComputeWorkloadAnalysis \
--section WarpStateStats --section Occupancy --section LaunchStats"

M="${M:-4096}"
N="${N:-4096}"
K="${K:-2048}"
DTYPES="${DTYPES:-fp16 fp8}"
for dt in $DTYPES; do
  for backend in tilelang tilelang_ws triton cutile; do
    name="${PREFIX:-m_}${backend}_${dt}"
    echo ">>> $name (M=$M N=$N K=$K, matched tile)"
    $NCU --profile-from-start off $SECTIONS \
         --target-processes all --force-overwrite \
         -o "reports/${name}" \
         $PY harness/profile_matmul.py --backend "$backend" --dtype "$dt" \
            --M "$M" --N "$N" --K "$K" --matched \
      2>&1 | tee "logs/${name}.log" | grep -E "max_abs_err|error|Error|==PROF== Disc" || true
  done
done
echo done
