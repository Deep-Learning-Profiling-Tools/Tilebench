"""NCU harness — cuTile block_sparse_attention, autotune-winner config.

Strategy: monkey-patch `_DEFAULT_CONFIG` to the cfg that real autotune
selected for the M=4096 case (occupancy=32), then run a clean launch
that NCU profiles.

Representative case: B=2, H=8, M=4096, D=128, BLOCK_M=BLOCK_N=64, BLOCK_D=128, fp16
(middle of the M sweep; autotune log records occupancy=32 as the winner).

Usage:
  ncu --set full --import-source on \\
      --launch-skip 3 --launch-count 1 \\
      --force-overwrite \\
      -o outputs/ncu/block_sparse_attention/cutile_autotune \\
      python tilebench/profiling/ncu_harness_block_sparse_attention_cutile.py
"""
import os
import sys

import torch
from types import SimpleNamespace
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.block_sparse_attention import impl_cutile

# Read M and autotune-winner occupancy from env so the same script can be
# reused across multiple problem sizes (autotune picks different cfgs).
M = int(os.environ.get("NCU_M", "4096"))
OCCUPANCY = int(os.environ.get("NCU_OCC", "32"))   # M=4096 winner
impl_cutile._DEFAULT_CONFIG = SimpleNamespace(occupancy=OCCUPANCY)

inputs = GENERATORS["block_sparse_attention"](
    B=2, H=8, M=M, D=128, H_kv=2,
    BLOCK_M=64, BLOCK_N=64, BLOCK_D=128, NUM_D_BLOCKS=1,
    dtype=torch.float16,
)

for _ in range(3):
    out = impl_cutile.run(*inputs)
    torch.cuda.synchronize()

out = impl_cutile.run(*inputs)
torch.cuda.synchronize()
print(f"cutile (autotune-cfg) out: shape={out.shape} dtype={out.dtype} "
      f"sum={out.float().sum().item():.4e}")
