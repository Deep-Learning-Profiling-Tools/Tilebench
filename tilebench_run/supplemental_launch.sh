#!/usr/bin/env bash
# Supplemental run: the 3 ops that crashed in the original sweep.
# Now fixed via the fix/2operator_bugs commits.
cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

OPS=(2d_conv linear_self_attention bitonic_sort)

bash tilebench_run/run_batch.sh supplemental "${OPS[@]}"
