#!/usr/bin/env bash
# Batch 1 — runs on the current interactive B200 session.
# 23 operators: slow + medium + fast mix; 2 timeout ops included.

cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

OPS=(
  # slow (8): matmul / conv / attention / norm — incl. 1 timeout (matmul_fp32_fp16_fp8)
  matmul_int8 matmul_fp32_fp16_fp8 streamk_matmul flash_attention block_sparse_attention 2d_conv layernorm l2_norm
  # medium (7): incl. 1 timeout (softmax)
  softmax cross_entropy argmax mean_reduction swiglu dropout destindex
  # fast (8)
  mul2 vector_add relu sigmoid matrix_transpose rope gaussian_blur bitonic_sort
)

bash tilebench/profiling/run_batch.sh batch1 "${OPS[@]}"
