import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _transpose_kernel(
    x,
    output,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    ALLOW_TMA: ConstBool,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    tile = ct.load(
        x,
        index=(bid_m, bid_n),
        shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=ALLOW_TMA,
    )
    out_tile = ct.transpose(tile)

    ct.store(
        output,
        index=(bid_n, bid_m),
        tile=out_tile,
        allow_tma=ALLOW_TMA,
    )


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")

    m = x.shape[0]
    n = x.shape[1]
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    occupancy = 4
    allow_tma = True

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(m, BLOCK_M), ct.cdiv(n, BLOCK_N), 1)
    kernel = _transpose_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, BLOCK_M, BLOCK_N, allow_tma))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "occupancy": occupancy,
            "allow_tma": allow_tma,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
