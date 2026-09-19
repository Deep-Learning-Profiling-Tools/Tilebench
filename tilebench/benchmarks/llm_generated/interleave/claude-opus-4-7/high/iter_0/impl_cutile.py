import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _interleave_kernel(A, B, output, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE
    offs = base + ct.arange(TILE, dtype=np.int32)
    a_tile = ct.load(A, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    in_bounds = offs < N
    # scatter a to even positions, b to odd positions
    ct.scatter(output, (ct.where(in_bounds, 2 * offs, -1),), a_tile)
    ct.scatter(output, (ct.where(in_bounds, 2 * offs + 1, -1),), b_tile)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _interleave_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (A, B, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
