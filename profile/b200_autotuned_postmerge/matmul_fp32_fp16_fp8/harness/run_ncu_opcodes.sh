#!/bin/bash
# Opcode histogram only. sass__inst_executed_per_opcode is a single-pass metric,
# so this runs in seconds where --section InstructionStats hangs.
set -e
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
mkdir -p reports logs

M="${M:-4096}"
N="${N:-4096}"
K="${K:-2048}"
DTYPES="${DTYPES:-fp16 fp8}"
for dt in $DTYPES; do
  for backend in tilelang tilelang_ws triton cutile; do
    name="o_${backend}_${dt}"
    echo ">>> $name (M=$M N=$N K=$K, matched tile, opcodes only)"
    $NCU --profile-from-start off \
         --metrics sass__inst_executed_per_opcode,smsp__inst_executed.sum \
         --target-processes all --force-overwrite \
         -o "reports/${name}" \
         $PY harness/profile_matmul.py --backend "$backend" --dtype "$dt" \
            --M "$M" --N "$N" --K "$K" --matched \
      2>&1 | tee "logs/${name}.log" | grep -E "max_abs_err|rror" || true
  done
done
echo done
