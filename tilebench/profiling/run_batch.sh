#!/usr/bin/env bash
# Run a list of operators in {default, autotune} mode.
# Per-op JSONs are renamed to ${op}_default.json / ${op}_autotune.json so both
# modes' results survive. Autotune mode for the 4 "problematic" ops has a
# 60-minute timeout (raised from 30 in the previous run); if it times out,
# we skip and continue.
#
# Usage: GPU=<label> run_batch.sh <batch_name> <op1> <op2> ...
#   GPU is the hardware label passed to run_bench.py --gpu (e.g. GPU=B200); it
#   selects the result namespace results/<GPU>/ and has no default.

set -u

GPU=${GPU:?set GPU to the hardware label of this machine, e.g. GPU=B200}
BATCH=${1:?batch name required}
shift
OPS=("$@")

# 4 ops with 60-min autotune timeout (others run without timeout)
TIMEOUT_OPS=(softmax matmul_fp32_fp16_fp8 kl_divergence histogramming)

REPO=/projects/kzhou6/bcui2/research/tilebench/Tilebench
LOG_DIR=${REPO}/outputs/${BATCH}
mkdir -p "$LOG_DIR"

cd "$REPO"

needs_timeout() {
  local op=$1
  for t in "${TIMEOUT_OPS[@]}"; do
    [ "$t" = "$op" ] && return 0
  done
  return 1
}

rename_outputs() {
  local op=$1
  local suffix=$2
  local src_json="results/${GPU}/logs/time_measurement_logs/${op}_results.json"
  local src_csv="results/${GPU}/csv/${op}_summary.csv"
  [ -f "$src_json" ] && cp "$src_json" "results/${GPU}/logs/time_measurement_logs/${op}_${suffix}.json"
  [ -f "$src_csv" ]  && cp "$src_csv"  "results/${GPU}/csv/${op}_${suffix}.csv"
}

START=$(date +%s)
echo "=== Batch ${BATCH} starting at $(date) on $(hostname) ==="
echo "Ops (${#OPS[@]}): ${OPS[*]}"
echo "Timeout list (60min): ${TIMEOUT_OPS[*]}"
echo

for op in "${OPS[@]}"; do
  echo "----- ${op} (default) -----"
  ts=$(date +%s)
  PYTHONPATH=. python scripts/run_bench.py --gpu "$GPU" --operator "$op" \
      > "$LOG_DIR/${op}_default.log" 2>&1
  rc=$?
  rename_outputs "$op" default
  echo "  default rc=$rc  elapsed=$(($(date +%s) - ts))s"

  echo "----- ${op} (autotune) -----"
  ts=$(date +%s)
  if needs_timeout "$op"; then
    timeout 3600 env PYTHONPATH=. python scripts/run_bench.py --gpu "$GPU" --operator "$op" --autotune \
        > "$LOG_DIR/${op}_autotune.log" 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then
      echo "  autotune TIMED OUT (60min) — skipped"
      echo "AUTOTUNE_TIMEOUT" > "$LOG_DIR/${op}_autotune.TIMEOUT"
      continue
    fi
  else
    PYTHONPATH=. python scripts/run_bench.py --gpu "$GPU" --operator "$op" --autotune \
        > "$LOG_DIR/${op}_autotune.log" 2>&1
    rc=$?
  fi
  rename_outputs "$op" autotune
  echo "  autotune rc=$rc  elapsed=$(($(date +%s) - ts))s"
done

echo
echo "=== Batch ${BATCH} done at $(date), total $(($(date +%s) - START))s ==="
