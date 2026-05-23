#!/usr/bin/env bash
# Re-run the original benchmark pipeline (scripts/run_bench.py) for a given
# list of operators on the *current* branch (TMA-descriptor refactor), in
# both default and autotune modes, and save outputs under
# results/descriptor_results/ mirroring the layout of results/.
#
# Layout produced:
#   results/descriptor_results/csv/<op>_default.csv
#   results/descriptor_results/csv/<op>_autotune.csv
#   results/descriptor_results/logs/time_measurement_logs/<op>_default.json
#   results/descriptor_results/logs/time_measurement_logs/<op>_autotune.json
#   results/descriptor_results/logs/autotune_logs/<op>_autotune.json
#
# Usage:
#   OPS="argmax batched_matmul ..." bash tilebench_run/descriptor_rerun.sh
#
# The CSV file that scripts/run_bench.py hard-codes to results/csv/<op>_summary.csv
# is moved into descriptor_results/csv/ after each mode finishes, so the
# canonical results/csv/ snapshot is never overwritten persistently.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT="results/descriptor_results"
mkdir -p \
    "$OUT/csv" \
    "$OUT/aggregate" \
    "$OUT/logs/time_measurement_logs" \
    "$OUT/logs/autotune_logs"

OPS="${OPS:-${*:-}}"
if [ -z "$OPS" ]; then
    echo "ERROR: pass ops via env (OPS='...') or args" >&2
    exit 1
fi

START=$(date +%s)
echo "============================================================"
echo "[desc] start $(date -Iseconds)  host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-?}"
echo "[desc] ops: $OPS"
echo "============================================================"

for op in $OPS; do
    op_start=$(date +%s)
    echo
    echo "============================================================"
    echo "[desc] $(date -Iseconds)  $op  (elapsed=$(((($(date +%s)-START))/60))min)"
    echo "============================================================"

    # ---- Default mode ----
    echo "--- $op DEFAULT ---"
    PYTHONPATH=. python -u scripts/run_bench.py --operator "$op" \
        --output "$OUT/logs/time_measurement_logs/${op}_default.json" \
        2>&1 | tail -30
    rc_def=$?
    if [ -f "results/csv/${op}_summary.csv" ]; then
        mv "results/csv/${op}_summary.csv" "$OUT/csv/${op}_default.csv"
    fi

    # ---- Autotune mode ----
    echo "--- $op AUTOTUNE ---"
    PYTHONPATH=. python -u scripts/run_bench.py --operator "$op" --autotune \
        --output "$OUT/logs/time_measurement_logs/${op}_autotune.json" \
        --autotune-log "$OUT/logs/autotune_logs/${op}_autotune.json" \
        2>&1 | tail -30
    rc_tune=$?
    if [ -f "results/csv/${op}_summary.csv" ]; then
        mv "results/csv/${op}_summary.csv" "$OUT/csv/${op}_autotune.csv"
    fi

    op_elapsed=$(($(date +%s) - op_start))
    echo "[desc] $op done in ${op_elapsed}s (rc default=$rc_def autotune=$rc_tune)"
done

ELAPSED=$(($(date +%s) - START))
echo
echo "============================================================"
echo "[desc] ALL DONE $(date -Iseconds)  total=$((ELAPSED/3600))h$((ELAPSED%3600/60))m"
echo "============================================================"
