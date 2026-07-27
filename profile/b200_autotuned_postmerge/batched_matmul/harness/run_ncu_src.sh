#!/bin/bash
source "$(dirname "$0")/env.sh"; cd "$RUNDIR"
for b in tilelang triton cutile; do
  echo ">>> src_${b}"
  $NCU --profile-from-start off --set source --section SourceCounters \
       --target-processes all --force-overwrite -o "reports/src_${b}_fp16" \
       $PY -u harness/profile_bmm.py --backend "$b" --dtype fp16 --M 384 \
    2>&1 | tee "logs/src_${b}.log" | grep -E "max_abs_err|Disc|rror" || true
  echo "<<< src_${b} done"
done
echo ALLDONE
