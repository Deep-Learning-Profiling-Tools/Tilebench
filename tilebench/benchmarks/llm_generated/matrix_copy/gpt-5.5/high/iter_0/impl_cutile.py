import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _matrix_copy_kernel(A, B, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(
        A,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        B,
        index=(bid,),
        tile=x_tile,
        latency=1,
        allow_tma=False,
    )


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    n_elements = A.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _matrix_copy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (A_flat, B_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return B


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
