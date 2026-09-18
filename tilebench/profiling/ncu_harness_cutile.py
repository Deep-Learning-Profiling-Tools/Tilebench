"""NCU profiling harness — cuTile matmul_int8.

Loads inputs, warms up the kernel (so the JIT cache + autotune state are
settled), then performs a single clean launch which NCU profiles.

Usage:
  ncu --set full --launch-skip 3 --launch-count 1 \
      -o outputs/ncu/matmul_int8_cutile \
      python tilebench/profiling/ncu_harness_cutile.py
"""
import sys

import torch
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.matmul_int8 import impl_cutile

# Match the default config used in the benchmark sweep
M, N, K = 2048, 2048, 4096
inputs = GENERATORS["matmul_int8"](M, N, K)

# 3 warmups so kernel + tuner caches are settled
for _ in range(3):
    out = impl_cutile.run(*inputs)
    torch.cuda.synchronize()

# Profiled launch
out = impl_cutile.run(*inputs)
torch.cuda.synchronize()
print(f"cutile out: shape={out.shape} dtype={out.dtype} sum={out.float().sum().item():.4e}")
