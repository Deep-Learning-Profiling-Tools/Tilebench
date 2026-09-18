import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _matmul_kernel(a, b, out,
                   K_TILES: ConstInt,
                   GRID_N: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    pid = ct.bid(0)

    pid_m = pid // GRID_N
    pid_n = pid - pid_m * GRID_N

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)

    for kk in range(0, K_TILES):
        a_tile = ct.load(
            a,
            index=(pid_m, kk),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(kk, pid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(out, index=(pid_m, pid_n), tile=ct.astype(acc, out.dtype))


@ct.kernel
def _matmul_kernel_tf32x9(a, b, out,
                          K_TILES: ConstInt,
                          GRID_N: ConstInt,
                          TM: ConstInt, TN: ConstInt, TK: ConstInt):
    pid = ct.bid(0)

    pid_m = pid // GRID_N
    pid_n = pid - pid_m * GRID_N

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)

    for kk in range(0, K_TILES):
        a_tile = ct.load(
            a,
            index=(pid_m, kk),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(kk, pid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        b0 = ct.astype(b_tile, ct.tfloat32)

        a0 = ct.astype(a_tile, ct.tfloat32)
        acc = ct.mma(a0, b0, acc)
        a_res0 = a_tile - ct.astype(a0, np.float32)
        a1 = ct.astype(a_res0, ct.tfloat32)
        acc = ct.mma(a1, b0, acc)
        a_res1 = a_res0 - ct.astype(a1, np.float32)
        a2 = ct.astype(a_res1, ct.tfloat32)
        acc = ct.mma(a2, b0, acc)

        b_res0 = b_tile - ct.astype(b0, np.float32)
        b1 = ct.astype(b_res0, ct.tfloat32)

        a0 = ct.astype(a_tile, ct.tfloat32)
        acc = ct.mma(a0, b1, acc)
        a_res0 = a_tile - ct.astype(a0, np.float32)
        a1 = ct.astype(a_res0, ct.tfloat32)
        acc = ct.mma(a1, b1, acc)
        a_res1 = a_res0 - ct.astype(a1, np.float32)
        a2 = ct.astype(a_res1, ct.tfloat32)
        acc = ct.mma(a2, b1, acc)

        b_res1 = b_res0 - ct.astype(b1, np.float32)
        b2 = ct.astype(b_res1, ct.tfloat32)

        a0 = ct.astype(a_tile, ct.tfloat32)
        acc = ct.mma(a0, b2, acc)
        a_res0 = a_tile - ct.astype(a0, np.float32)
        a1 = ct.astype(a_res0, ct.tfloat32)
        acc = ct.mma(a1, b2, acc)
        a_res1 = a_res0 - ct.astype(a1, np.float32)
        a2 = ct.astype(a_res1, ct.tfloat32)
        acc = ct.mma(a2, b2, acc)

    ct.store(out, index=(pid_m, pid_n), tile=ct.astype(acc, out.dtype))


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    TM = 128
    TN = 128
    TK = 64
    K_TILES = (k + TK - 1) // TK
    GRID_M = (m + TM - 1) // TM
    GRID_N = (n + TN - 1) // TN
    occupancy = 1

    grid = (GRID_M * GRID_N, 1, 1)
    stream = torch.cuda.current_stream()

    if a.dtype == torch.float32:
        kernel = _matmul_kernel_tf32x9.with_hints(occupancy=occupancy)
        ct.launch(stream, grid, kernel, (a, b, output, K_TILES, GRID_N, TM, TN, TK))
        fp32_mode = "tf32x9"
    else:
        kernel = _matmul_kernel.with_hints(occupancy=occupancy)
        ct.launch(stream, grid, kernel, (a, b, output, K_TILES, GRID_N, TM, TN, TK))
        fp32_mode = "n/a"

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM,
        "TN": TN,
        "TK": TK,
        "K_TILES": K_TILES,
        "GRID_M": GRID_M,
        "GRID_N": GRID_N,
        "occupancy": occupancy,
        "fp32_input": fp32_mode,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
