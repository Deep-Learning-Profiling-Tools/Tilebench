"""NCU profiling harness — Triton matmul_int8.

Loads inputs, warms up the kernel (JIT compile + Triton autotune cache),
then performs a single clean launch which NCU profiles.

Usage:
  ncu --set full --launch-skip 3 --launch-count 1 \
      -o outputs/ncu/<gpu>/matmul_int8_triton \
      python tilebench/profiling/ncu_harness_triton.py
"""
import sys

import torch
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.matmul_int8 import impl_triton

# Match the default config used in the benchmark sweep
M, N, K = 2048, 2048, 4096
inputs = GENERATORS["matmul_int8"](M, N, K)

# 3 warmups so kernel JIT + Triton autotune cache are settled
for _ in range(3):
    out = impl_triton.run(*inputs)
    torch.cuda.synchronize()

# Profiled launch
out = impl_triton.run(*inputs)
torch.cuda.synchronize()
print(f"triton out: shape={out.shape} dtype={out.dtype} sum={out.float().sum().item():.4e}")
