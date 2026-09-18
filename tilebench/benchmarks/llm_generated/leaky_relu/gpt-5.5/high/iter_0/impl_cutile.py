import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _leaky_relu_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    neg = ct.astype(ct.astype(x_tile, np.float32) * 0.01, x.dtype)
    y_tile = ct.where(x_tile > 0.0, x_tile, neg)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _leaky_relu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (input, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
