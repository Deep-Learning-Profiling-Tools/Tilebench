"""NCU harness — Triton matmul_int8, using the autotune-WINNING config
directly (no sweep at profile time).

Strategy: skip autotune entirely. Override `_DEFAULT_CONFIG` to the
config that real autotune selected previously (recorded by the sanity
run). NCU profiles a clean single launch of the winner.

Sanity run had picked:
  BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
  GROUP_SIZE_M=8, num_warps=4, num_stages=3.

Usage:
  ncu --set full --import-source on \
      --kernel-name regex:"matmul" \
      --launch-skip 3 --launch-count 1 \
      --force-overwrite \
      -o outputs/ncu/matmul_int8_triton_autotune \
      python tilebench/profiling/ncu_harness_triton_autotune.py
"""
import sys

import torch
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.matmul_int8 import impl_triton

# Override _DEFAULT_CONFIG to match what autotune picked previously
impl_triton._DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps":    4,
    "num_stages":   3,
}

M, N, K = 2048, 2048, 4096
inputs = GENERATORS["matmul_int8"](M, N, K)

for _ in range(3):
    out = impl_triton.run(*inputs)
    torch.cuda.synchronize()

out = impl_triton.run(*inputs)
torch.cuda.synchronize()
print(f"triton (autotune-cfg) out: shape={out.shape} dtype={out.dtype} sum={out.float().sum().item():.4e}")
