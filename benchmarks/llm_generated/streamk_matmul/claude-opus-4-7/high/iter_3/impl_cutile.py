import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _splitk_lowp(a, b, c,
                 ITERS_PER_SPLIT: ConstInt,
                 BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    split_id = ct.bid(2)
    k_base = split_id * ITERS_PER_SPLIT

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m[:, None], offs_n[None, :]), acc)


@ct.kernel(occupancy=1)
def _splitk_fp32_hp(a, b, c,
                    ITERS_PER_SPLIT: ConstInt,
                    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    split_id = ct.bid(2)
    k_base = split_id * ITERS_PER_SPLIT

    # Promote to fp64 internally: ct.mma(fp32) uses TF32 cores which loses
    # precision vs the reference; fp64 mma gives well-above-fp32 accuracy.
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float64)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        a_d = ct.astype(a_tile, np.float64)
        b_d = ct.astype(b_tile, np.float64)
        acc = ct.mma(a_d, b_d, acc)

    acc_f32 = ct.astype(acc, np.float32)
    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m[:, None], offs_n[None, :]), acc_f32)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    if a.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        ITERS_PER_SPLIT = 32
        kernel = _splitk_fp32_hp
    else:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        ITERS_PER_SPLIT = 32
        kernel = _splitk_lowp

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    iters_per_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_splits = (iters_per_tile + ITERS_PER_SPLIT - 1) // ITERS_PER_SPLIT
    total_tiles = num_pid_m * num_pid_n

    stream = torch.cuda.current_stream()
    grid = (num_pid_m, num_pid_n, num_splits)
    ct.launch(
        stream, grid, kernel,
        (a, b, output_fp32, ITERS_PER_SPLIT, BLOCK_M, BLOCK_N, BLOCK_K),
    )

    if a.dtype == torch.float32:
        output = output_fp32
    else:
        output = output_fp32.to(a.dtype)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "ITERS_PER_SPLIT": ITERS_PER_SPLIT,
        "num_splits": num_splits,
        "total_tiles": total_tiles,
        "fp32_path": "fp64_internal" if a.dtype == torch.float32 else "native",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
