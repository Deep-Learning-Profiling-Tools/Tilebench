import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _matmul_lowp(a, b, c, K_TILES: ConstInt,
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


@ct.kernel(occupancy=2)
def _matmul_fp32_3xtf32(a, b, c, K_TILES: ConstInt,
                        BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)

        # 3xTF32 emulation: split fp32 into hi (truncated to tf32 precision)
        # and lo (residual, also tf32-representable). Then 3 mmas approximate
        # fp32 to ~10^-7 absolute, well within fp32 tolerance.
        a_hi = ct.astype(a_tile, ct.tfloat32)
        a_hi_f = ct.astype(a_hi, np.float32)
        a_lo = ct.astype(a_tile - a_hi_f, ct.tfloat32)

        b_hi = ct.astype(b_tile, ct.tfloat32)
        b_hi_f = ct.astype(b_hi, np.float32)
        b_lo = ct.astype(b_tile - b_hi_f, ct.tfloat32)

        acc = ct.mma(a_hi, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)
        acc = ct.mma(a_lo, b_hi, acc)

    ct.store(c, index=(pid_m, pid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    K_TILES = (K + BLOCK_K - 1) // BLOCK_K
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    stream = torch.cuda.current_stream()
    grid = (num_pid_m, num_pid_n, 1)

    if a.dtype == torch.float32:
        kernel = _matmul_fp32_3xtf32
        kernel_name = "fp32_3xtf32"
    else:
        kernel = _matmul_lowp
        kernel_name = "lowp"

    ct.launch(stream, grid, kernel,
              (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "K_TILES": K_TILES,
        "num_pid_m": num_pid_m, "num_pid_n": num_pid_n,
        "kernel": kernel_name,
        "occupancy": 2,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
