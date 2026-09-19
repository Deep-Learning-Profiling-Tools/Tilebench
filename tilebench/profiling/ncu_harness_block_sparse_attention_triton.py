"""NCU harness — Triton block_sparse_attention, autotune-winner config.

For the M=4096 case, autotune picks num_warps=4, num_stages=2, which
is also the current `_DEFAULT_CONFIG` value — no override needed.
Set explicitly anyway for clarity.

Representative case: B=2, H=8, M=4096, D=128, BLOCK_M=BLOCK_N=64, BLOCK_D=128, fp16.

Usage:
  ncu --set full --import-source on \\
      --launch-skip 3 --launch-count 1 \\
      --force-overwrite \\
      -o outputs/ncu/<gpu>/block_sparse_attention/triton_autotune \\
      python tilebench/profiling/ncu_harness_block_sparse_attention_triton.py
"""
import os
import sys

import torch
from tilebench.data.tensors import GENERATORS
from tilebench.benchmarks.operators.block_sparse_attention import impl_triton

# Triton winner is num_warps=4 / num_stages=2 at every M in the sweep
# (only M=512 picks num_warps=8) — keep these as default-overridable env vars.
M = int(os.environ.get("NCU_M", "4096"))
NUM_WARPS = int(os.environ.get("NCU_WARPS", "4"))
NUM_STAGES = int(os.environ.get("NCU_STAGES", "2"))
impl_triton._DEFAULT_CONFIG = {"num_warps": NUM_WARPS, "num_stages": NUM_STAGES}

inputs = GENERATORS["block_sparse_attention"](
    B=2, H=8, M=M, D=128, H_kv=2,
    BLOCK_M=64, BLOCK_N=64, BLOCK_D=128, NUM_D_BLOCKS=1,
    dtype=torch.float16,
)

for _ in range(3):
    out = impl_triton.run(*inputs)
    torch.cuda.synchronize()

out = impl_triton.run(*inputs)
torch.cuda.synchronize()
print(f"triton (autotune-cfg) out: shape={out.shape} dtype={out.dtype} "
      f"sum={out.float().sum().item():.4e}")
