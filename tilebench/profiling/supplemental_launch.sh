#!/usr/bin/env bash
# Supplemental run: the 3 ops that crashed in the original sweep.
# Now fixed via the fix/2operator_bugs commits.
# Usage: GPU=<label> tilebench/profiling/supplemental_launch.sh   (from any directory)

OPS=(2d_conv linear_self_attention bitonic_sort)

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_batch.sh" supplemental "${OPS[@]}"
