#!/bin/bash
# Formal TileBench++ GH200 campaign driver (lives outside the repo). One run_bench.py process per
# operator, strictly sequential, official CLI only. Usage: run_campaign.sh default|autotune [ops...]
set -u
MODE=$1; shift
W=/projects/kzhou6/bcui2/research/tilebench/Tilebench-gh200; C=/tmp/tilebench-gh200-campaign/$MODE
cd $W; source /projects/kzhou6/bcui2/env_software/miniconda3-aarch64/etc/profile.d/conda.sh
conda activate "tilebench++_env"; export PYTHONPATH=.
mkdir -p $C/console
EXPECT=${EXPECT:?}
[ "$(git rev-parse HEAD)" = "$EXPECT" ] || { echo "ABORT: HEAD $(git rev-parse HEAD) != $EXPECT"; exit 2; }
[ -z "$(git status --short --untracked-files=no)" ] || { echo "ABORT: tracked modifications"; exit 2; }
EXTRA=(); TMO=10800; [ "$MODE" = autotune ] && { EXTRA=(--autotune); TMO=86400; }
OPS=${*:-$(python -c "import json;print(' '.join(json.load(open('/tmp/tilebench-gh200-campaign/selection.json'))))")}
echo "START $(date -Is) mode=$MODE head=$(git rev-parse HEAD) ops=$(echo $OPS | wc -w)"
for op in $OPS; do
  s=$(date +%s)
  timeout -k 120 $TMO python scripts/run_bench.py --gpu GH200 --operator "$op" \
      --tile-language triton,cutile,tilelang --warmup 1 --repeat 3 "${EXTRA[@]}" > "$C/console/$op.log" 2>&1
  rc=$?
  printf "%s\t%s\t%ss\t%s\n" "$op" "$rc" "$(( $(date +%s) - s ))" "$(date -Is)" | tee -a $C/exit_codes.tsv
done
echo "DONE $(date -Is) mode=$MODE"
