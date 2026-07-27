#!/bin/bash
# Shared environment for this profiling run.
export PYTHONPATH=/home/arustagi/repos/Tilebench
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++'
# SLURM job 9073602 was allocated GPU IDX:0 on dgx003; pin by UUID.
export CUDA_VISIBLE_DEVICES=GPU-76fdb436-684f-79c7-5f3b-f742c09ba8c9
export PY=/home/arustagi/anaconda3/envs/tilebench/bin/python
export NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
export RUNDIR=/home/arustagi/repos/Tilebench/profile/b200_autotuned_postmerge/matmul_fp32_fp16_fp8
