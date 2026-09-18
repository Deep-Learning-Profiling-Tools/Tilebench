import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _transpose_kernel(x, output, BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    tile = ct.load(
        x,
        index=(bid_m, bid_n),
        shape=(BLOCK_M, BLOCK_N),
        latency=1,
        allow_tma=False,
    )
    out_tile = ct.transpose(tile)

    ct.store(
        output,
        index=(bid_n, bid_m),
        tile=out_tile,
        latency=1,
        allow_tma=False,
    )


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")

    m = x.shape[0]
    n = x.shape[1]
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 32
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(m, BLOCK_M), ct.cdiv(n, BLOCK_N), 1)
    ct.launch(stream, grid, _transpose_kernel, (x, output, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "occupancy": occupancy,
            "allow_tma": False,
            "latency": 1,
            "padding_mode": "UNDETERMINED",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
