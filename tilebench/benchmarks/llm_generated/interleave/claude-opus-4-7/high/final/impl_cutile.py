import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _interleave_kernel(A, B, output, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    a_tile = ct.load(A, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    # Interleave: stack as (TILE, 2) then flatten -> [a0,b0,a1,b1,...]
    a_2d = ct.expand_dims(a_tile, axis=1)   # (TILE, 1)
    b_2d = ct.expand_dims(b_tile, axis=1)   # (TILE, 1)
    ab = ct.cat((a_2d, b_2d), axis=1)       # (TILE, 2)
    out_tile = ct.reshape(ab, (2 * TILE,))
    # OOB stores silently dropped for the tail tile
    ct.store(output, index=(bid,), tile=out_tile)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _interleave_kernel, (A, B, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
