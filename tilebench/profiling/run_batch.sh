#!/usr/bin/env bash
# Run a list of operators in {default, autotune} mode on one GPU.
#
# run_bench.py owns the result names: every run writes its raw JSON to
#   results/<GPU>/logs/{time_measurement_logs,autotune_logs}/<op>_<mode>_<backends>.json
# and its summary to results/<GPU>/csv/<op>_<mode>.csv, so the default and the
# autotune pass of one operator never overwrite each other and nothing is
# renamed or copied here. Autotune mode for the 4 "problematic" ops has a
# 60-minute timeout (raised from 30 in the previous run); if it times out,
# we skip and continue.
#
# Usage: GPU=<label> run_batch.sh <batch_name> <op1> <op2> ...
#   GPU is the hardware label passed to run_bench.py --gpu (e.g. GPU=B200); it
#   selects the result namespace results/<GPU>/ and has no default.
#   Works from any directory: the repository is located from this file.

set -u

GPU=${GPU:?set GPU to the hardware label of this machine, e.g. GPU=B200}
BATCH=${1:?batch name required}
shift
OPS=("$@")

# 4 ops with 60-min autotune timeout (others run without timeout)
TIMEOUT_OPS=(softmax matmul_fp32_fp16_fp8 kl_divergence histogramming)

# tilebench/profiling/run_batch.sh -> repository root
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_BENCH="$REPO/scripts/run_bench.py"
LOG_DIR=${REPO}/outputs/${BATCH}
mkdir -p "$LOG_DIR"

needs_timeout() {
  local op=$1
  for t in "${TIMEOUT_OPS[@]}"; do
    [ "$t" = "$op" ] && return 0
  done
  return 1
}

START=$(date +%s)
echo "=== Batch ${BATCH} on ${GPU} starting at $(date) on $(hostname) ==="
echo "Ops (${#OPS[@]}): ${OPS[*]}"
echo "Timeout list (60min): ${TIMEOUT_OPS[*]}"
echo

for op in "${OPS[@]}"; do
  echo "----- ${op} (default) -----"
  ts=$(date +%s)
  python "$RUN_BENCH" --gpu "$GPU" --operator "$op" \
      > "$LOG_DIR/${op}_default.log" 2>&1
  rc=$?
  echo "  default rc=$rc  elapsed=$(($(date +%s) - ts))s"

  echo "----- ${op} (autotune) -----"
  ts=$(date +%s)
  if needs_timeout "$op"; then
    timeout 3600 python "$RUN_BENCH" --gpu "$GPU" --operator "$op" --autotune \
        > "$LOG_DIR/${op}_autotune.log" 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then
      echo "  autotune TIMED OUT (60min) — skipped"
      echo "AUTOTUNE_TIMEOUT" > "$LOG_DIR/${op}_autotune.TIMEOUT"
      continue
    fi
  else
    python "$RUN_BENCH" --gpu "$GPU" --operator "$op" --autotune \
        > "$LOG_DIR/${op}_autotune.log" 2>&1
    rc=$?
  fi
  echo "  autotune rc=$rc  elapsed=$(($(date +%s) - ts))s"
done

echo
echo "=== Batch ${BATCH} done at $(date), total $(($(date +%s) - START))s ==="
