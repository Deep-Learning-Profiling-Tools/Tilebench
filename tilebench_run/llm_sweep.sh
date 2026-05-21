#!/usr/bin/env bash
# Sweep all 44 operators (45 minus vector_add — already piloted) through
# tools/llm_codegen/generate.py for one (model, effort) pair.
#
# Usage:
#   bash tilebench_run/llm_sweep.sh <model> <effort> [max_iters] [threshold]
#
# Per-op wall-clock cap is enforced with `timeout` so a single bad op
# (autotune hang, LLM API stall) can't soak the whole sweep budget.
set -uo pipefail

MODEL="${1:?usage: $0 <model> <effort> [max_iters] [threshold]}"
EFFORT="${2:?usage: $0 <model> <effort> [max_iters] [threshold]}"
MAX_ITERS="${3:-10}"
THRESHOLD="${4:-0.80}"

# 3 hours per op cap. Worst case 44 ops × 3 h = 132 h, but in practice
# easy ops freeze at iter_0 in ~5 min and only the hard L4/L5 ops use
# the full budget. Adjust if needed.
PER_OP_CAP_S=10800

# Skip operators where data generation or the framework is incompatible
# with the LLM pipeline (e.g. _template placeholder, vector_add piloted).
SKIP=" vector_add _template "

OPS=$(ls benchmarks/operators/ | grep -v "\.md$\|BENCHMARK")

START_T=$(date +%s)
echo "============================================================"
echo "[sweep] start  model=$MODEL effort=$EFFORT max_iters=$MAX_ITERS threshold=$THRESHOLD"
echo "[sweep] per-op cap=${PER_OP_CAP_S}s"
echo "[sweep] start time: $(date -Iseconds)"
echo "============================================================"

for op in $OPS; do
    if [[ "$SKIP" == *" $op "* ]]; then
        echo "[sweep] skip $op"
        continue
    fi
    echo
    echo "============================================================"
    echo "[sweep] $(date -Iseconds)  $MODEL/$EFFORT/$op  (elapsed=$((($(date +%s) - START_T)/60))min)"
    echo "============================================================"
    timeout "${PER_OP_CAP_S}" python -u tools/llm_codegen/generate.py \
        --operator "$op" --model "$MODEL" --effort "$EFFORT" \
        --max-iters "$MAX_ITERS" --threshold "$THRESHOLD"
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "[sweep] $op TIMEOUT after ${PER_OP_CAP_S}s"
    elif [ $rc -ne 0 ]; then
        echo "[sweep] $op FAILED rc=$rc"
    fi
done

ELAPSED=$(( $(date +%s) - START_T ))
echo
echo "============================================================"
echo "[sweep] DONE  model=$MODEL effort=$EFFORT  total=$((ELAPSED/3600))h$(((ELAPSED%3600)/60))m"
echo "============================================================"
