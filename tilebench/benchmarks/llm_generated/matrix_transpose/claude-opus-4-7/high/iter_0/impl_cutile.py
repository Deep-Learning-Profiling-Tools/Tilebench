import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _transpose_kernel(x, output, BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    tile = ct.load(x, index=(bid_m, bid_n), shape=(BM, BN),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_t = ct.transpose(tile)
    ct.store(output, index=(bid_n, bid_m), tile=tile_t)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream()

    BM = 64
    BN = 64
    occupancy = 4

    grid = (ct.cdiv(M, BM), ct.cdiv(N, BN), 1)
    kernel = _transpose_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
