import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _matmul_kernel_lowp(a, b, c, K_TILES: ConstInt,
                         BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    result = ct.astype(acc, c.dtype)
    ct.store(c, index=(pid_m, pid_n), tile=result)


@ct.kernel(occupancy=1)
def _matmul_kernel_fp32_fp64(a, b, c, K_TILES: ConstInt,
                              BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    # fp32 inputs upcast to fp64 in registers, then fp64 ct.mma → IEEE-grade.
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float64)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        a64 = ct.astype(a_tile, np.float64)
        b64 = ct.astype(b_tile, np.float64)
        acc = ct.mma(a64, b64, acc)

    result = ct.astype(acc, c.dtype)
    ct.store(c, index=(pid_m, pid_n), tile=result)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    stream = torch.cuda.current_stream()

    if a.dtype == torch.float32:
        # IEEE-fp32 via fp64 tensor cores (B200 supports fp64 ct.mma).
        # Small tile to keep fp64 register pressure manageable.
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        K_TILES = (K + BLOCK_K - 1) // BLOCK_K
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        grid = (num_pid_m, num_pid_n, 1)
        ct.launch(stream, grid, _matmul_kernel_fp32_fp64,
                  (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))
        kernel_kind = "fp64_via_mma"
        occ = 1
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        K_TILES = (K + BLOCK_K - 1) // BLOCK_K
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        grid = (num_pid_m, num_pid_n, 1)
        ct.launch(stream, grid, _matmul_kernel_lowp,
                  (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))
        kernel_kind = "lowp_mma"
        occ = 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "K_TILES": K_TILES,
        "num_pid_m": num_pid_m, "num_pid_n": num_pid_n,
        "kernel": kernel_kind,
        "occupancy": occ,
        "dtype": str(a.dtype),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
