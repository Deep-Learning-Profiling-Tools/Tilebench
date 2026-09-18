import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _relu_kernel_exact(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        allow_tma=False,
    )

    # Torch ReLU/threshold semantics: x <= 0 becomes +0, NaNs pass through.
    y_tile = ct.where(x_tile <= 0, 0, x_tile)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


@ct.kernel
def _relu_kernel_padded(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )

    # Torch ReLU/threshold semantics: x <= 0 becomes +0, NaNs pass through.
    y_tile = ct.where(x_tile <= 0, 0, x_tile)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8
    allow_tma = False

    use_exact_kernel = (n_elements % TILE) == 0
    if use_exact_kernel:
        grid = (n_elements // TILE, 1, 1)
        kernel = _relu_kernel_exact.with_hints(occupancy=occupancy)
    else:
        grid = (ct.cdiv(n_elements, TILE), 1, 1)
        kernel = _relu_kernel_padded.with_hints(occupancy=occupancy)

    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "allow_tma": allow_tma,
        "tail_padding": not use_exact_kernel,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
