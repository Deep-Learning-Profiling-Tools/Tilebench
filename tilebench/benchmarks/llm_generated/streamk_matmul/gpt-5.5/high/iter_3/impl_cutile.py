import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _matmul_kernel(a, b, out,
                   K_TILES: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    pid = ct.bid(0)

    num_pid_n = ct.num_tiles(out, 1, (TM, TN))
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

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

        if a.dtype == ct.float32:
            a_tile = ct.astype(a_tile, ct.tfloat32)
            b_tile = ct.astype(b_tile, ct.tfloat32)

        acc = ct.mma(a_tile, b_tile, acc)

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
    occupancy = 1

    grid_m = (m + TM - 1) // TM
    grid_n = (n + TN - 1) // TN
    grid = (grid_m * grid_n, 1, 1)

    stream = torch.cuda.current_stream()
    ct.launch(stream, grid, _matmul_kernel, (a, b, output, K_TILES, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM,
        "TN": TN,
        "TK": TK,
        "K_TILES": K_TILES,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
