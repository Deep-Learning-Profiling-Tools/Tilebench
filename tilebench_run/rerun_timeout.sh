#!/usr/bin/env bash
# v5: after both hang-cfg filter AND search-space shrinking.
# Expected per-op time: softmax ~3min, matmul ~15min, kl ~1min, hist ~5min.
# 30-min cap each is plenty.

set -u
cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

OPS=(softmax matmul_fp32_fp16_fp8 kl_divergence histogramming)
RUN_DIR=tilebench_run/autotune_timing_rerun
SUMMARY="$RUN_DIR/timing_summary.txt"

echo "" >> "$SUMMARY"
echo "=== v5 (filtered + shrunk, 30min cap): $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for op in "${OPS[@]}"; do
  out="$RUN_DIR/${op}_v5.out"
  echo "=========================================================" | tee -a "$SUMMARY"
  echo "[$(date -u +%H:%M:%SZ)] v5 starting $op autotune ..."      | tee -a "$SUMMARY"
  echo "=========================================================" | tee -a "$SUMMARY"
  t0=$(date +%s)
  timeout 1800 \
    bash -c "PYTHONPATH=. python -u scripts/run_bench.py \
      --operator '$op' --warmup 20 --repeat 100 --autotune" \
      > "$out" 2>&1
  rc=$?
  t1=$(date +%s)
  dur=$((t1 - t0))
  mins=$((dur / 60))
  secs=$((dur % 60))
  echo "[$(date -u +%H:%M:%SZ)] v5 $op finished rc=$rc, took ${mins}m ${secs}s" \
      | tee -a "$SUMMARY"
done

echo ""                                                  | tee -a "$SUMMARY"
echo "=== v5 End: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="    | tee -a "$SUMMARY"
