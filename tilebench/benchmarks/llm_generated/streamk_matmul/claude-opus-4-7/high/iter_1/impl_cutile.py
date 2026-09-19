import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _splitk_kernel(a, b, c, num_pid_n,
                   ITERS_PER_SPLIT: ConstInt,
                   BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    tile_id = ct.bid(0)
    split_id = ct.bid(1)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    k_base = split_id * ITERS_PER_SPLIT

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    # Atomic add the fp32 accumulator to the fp32 intermediate buffer.
    offs_m_idx = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n_idx = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m_idx[:, None], offs_n_idx[None, :]), acc)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    # Always accumulate into fp32 buffer; cast once at the end.
    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    ITERS_PER_SPLIT = 32

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    iters_per_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_splits = (iters_per_tile + ITERS_PER_SPLIT - 1) // ITERS_PER_SPLIT
    total_tiles = num_pid_m * num_pid_n

    stream = torch.cuda.current_stream()
    grid = (total_tiles, num_splits, 1)
    ct.launch(
        stream, grid, _splitk_kernel,
        (a, b, output_fp32, num_pid_n, ITERS_PER_SPLIT, BLOCK_M, BLOCK_N, BLOCK_K),
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
        "occupancy": 2,
        "num_splits": num_splits,
        "total_tiles": total_tiles,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
