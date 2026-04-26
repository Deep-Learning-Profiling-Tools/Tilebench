"""cuTile generic GEMM. Supports fp32 (TF32-accelerated via explicit cast),
fp16, fp8 e4m3fn, fp8 e5m2 — same kernel, output dtype follows C's torch dtype.

Algorithm: 1D grid with grouped scheduling (L2 cache swizzling); fp32
accumulator via ct.mma; cast to output dtype at the end.

Note on TF32: cuTile does NOT auto-promote fp32 inputs to TF32 inside
ct.mma — without an explicit cast, fp32 falls off the Tensor Core path.
To use Tensor Cores for fp32, the loaded tiles are cast via
`.astype(ct.tfloat32)` before ct.mma. This matches NVIDIA's official
MatMul.py sample in the cuda-tile-python repo. For fp16 / fp8 the cast
is a no-op (cast to same dtype) so it is free to apply unconditionally.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

# Per-dtype default tile sizes.
#   fp32 — Blackwell's fp32 Tensor Core MMA shape is small (~16x16x8); per
#          NVIDIA's cuTile matmul tutorial, fp32 matmul wants tm=tn=tk=32.
#          Larger fp32 tiles fall off the Tensor Core path and run scalar
#          (orders-of-magnitude slower).
#   fp16 / fp8 — large tiles fit HMMA / IMMA Tensor Core natively.
_DEFAULT_CONFIGS = {
    torch.float32:        SimpleNamespace(tm=32,  tn=32,  tk=32,  group_size_m=8, occupancy=4),
    torch.float16:        SimpleNamespace(tm=128, tn=256, tk=64,  group_size_m=8, occupancy=4),
    torch.float8_e4m3fn:  SimpleNamespace(tm=128, tn=256, tk=128, group_size_m=8, occupancy=4),
    torch.float8_e5m2:    SimpleNamespace(tm=128, tn=256, tk=128, group_size_m=8, occupancy=4),
}

# Search space (independent sweep, mirroring Triton). exhaustive_search
# silently skips configs the cuTile compiler rejects.
# Note: (tm=32, tn=32) is intentionally excluded — fp32's small-tile sweet
# spot (32x32x32 per NVIDIA's cuTile guide) is therefore not reachable
# from autotune, only from _DEFAULT_CONFIGS.
_SEARCH_SPACE = [
    SimpleNamespace(tm=tm, tn=tn, tk=tk, group_size_m=gs, occupancy=occ)
    for tm in [64, 128, 256]
    for tn in [128, 256]
    for tk in [32, 64, 128]
    for gs in [4, 8, 16]
    for occ in [4, 8, 16]
]


@ct.kernel
def matmul_kernel(
    A, B, C,
    M, N,
    K: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    pid = ct.bid(0)
    num_pid_m = ct.cdiv(M, TM)
    num_pid_n = ct.cdiv(N, TN)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = ct.zeros((TM, TN), dtype=ct.float32)
    num_tiles_k = ct.cdiv(K, TK)

    # Convert fp32 inputs to tf32 to use Tensor Cores; pass-through (no-op
    # cast to same dtype) for fp16 / fp8.
    mma_dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        b_tile = ct.load(B, index=(k, pid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(matmul_kernel)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    """cuTile matmul. Output dtype matches input dtype."""
    global _last_autotune_config

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(M, N, K, str(a.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                ((M + cfg.tm - 1) // cfg.tm) * ((N + cfg.tn - 1) // cfg.tn),
                1, 1,
            ),
            args_fn=lambda cfg: (
                a, b, c, M, N, K, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {
            "tm":           cfg.tm,
            "tn":           cfg.tn,
            "tk":           cfg.tk,
            "group_size_m": cfg.group_size_m,
            "occupancy":    cfg.occupancy,
        }
    else:
        if a.dtype not in _DEFAULT_CONFIGS:
            raise ValueError(f"No default config for dtype {a.dtype}")
        cfg = _DEFAULT_CONFIGS[a.dtype]

    grid = (
        ((M + cfg.tm - 1) // cfg.tm) * ((N + cfg.tn - 1) // cfg.tn),
        1, 1,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (a, b, c, M, N, K, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m),
    )
    return c


def get_last_config() -> dict | None:
    return _last_autotune_config
