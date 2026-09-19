"""NCU harness — cuTile matmul_int8, using the autotune-WINNING config
directly (no sweep at profile time).

Strategy: skip autotune entirely. Run the same flow as the default
harness, but monkey-patch `_DEFAULT_CONFIG` to the config that the
real autotune previously selected (recorded by the sanity run).
This way NCU profiles a clean single launch of the winner.

Sanity run had picked: tm=256, tn=64, tk=64, group_size_m=8, occupancy=4.

Usage:
  ncu --set full --import-source on \
      --kernel-name regex:"matmul" \
      --launch-skip 3 --launch-count 1 \
      --force-overwrite \
      -o outputs/ncu/<gpu>/matmul_int8_cutile_autotune \
      python tilebench/profiling/ncu_harness_cutile_autotune.py
"""
import sys

import torch
from types import SimpleNamespace
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.matmul_int8 import impl_cutile

# Override _DEFAULT_CONFIG to match what autotune picked previously
impl_cutile._DEFAULT_CONFIG = SimpleNamespace(
    tm=256, tn=64, tk=64, group_size_m=8, occupancy=4
)

M, N, K = 2048, 2048, 4096
inputs = GENERATORS["matmul_int8"](M, N, K)

# 3 warmups for JIT/cache, NCU profiles the 4th launch
for _ in range(3):
    out = impl_cutile.run(*inputs)
    torch.cuda.synchronize()

out = impl_cutile.run(*inputs)
torch.cuda.synchronize()
print(f"cutile (autotune-cfg) out: shape={out.shape} dtype={out.dtype} sum={out.float().sum().item():.4e}")
