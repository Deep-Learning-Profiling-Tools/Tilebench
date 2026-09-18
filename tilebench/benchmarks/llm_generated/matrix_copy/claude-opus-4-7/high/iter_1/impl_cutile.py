import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

TILE = 4096
OCCUPANCY = 4


@ct.kernel(occupancy=OCCUPANCY)
def _copy_kernel(x, out, TILE_C: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(bid,), tile=x_tile)


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    n_elements = A.numel()
    stream = torch.cuda.current_stream()

    A_flat = A.view(-1)
    B_flat = B.view(-1)

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _copy_kernel, (A_flat, B_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": OCCUPANCY})
    return B


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
