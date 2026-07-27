#!/bin/bash
# Collect --set full (+PM sampling) and --set source profiles for all three
# backends at both shapes. One kernel launch is captured per invocation
# (the harness brackets it with cudaProfilerStart/Stop).
set -e
source "$(dirname "$0")/env.sh"
cd "$RUNDIR"
mkdir -p reports logs

for backend in tilelang triton cutile; do
  for n in 4096 1048576; do
    case $n in 4096) tag=lowN ;; *) tag=highN ;; esac
    name="${backend}_${tag}"

    echo "=== full: $name ==="
    $NCU --profile-from-start off \
         --set full \
         --section PmSampling \
         --section PmSampling_WarpStates \
         --target-processes all \
         --force-overwrite \
         -o "reports/full_${name}" \
         $PY harness/profile_topk.py --backend "$backend" --n "$n" \
         > "logs/full_${name}.log" 2>&1

    echo "=== source: $name ==="
    $NCU --profile-from-start off \
         --set source \
         --section SourceCounters \
         --target-processes all \
         --force-overwrite \
         -o "reports/source_${name}" \
         $PY harness/profile_topk.py --backend "$backend" --n "$n" \
         > "logs/source_${name}.log" 2>&1
  done
done
echo done
