import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _dropout_kernel(x, x_keep, output, scale: float, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(x_keep, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    # Multiply natively in input dtype — avoids fp16/bf16 -> fp32 -> fp16/bf16
    # conversion overhead. scale (Python float) auto-promotes to tile dtype.
    y = x_tile * k_tile * scale
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8
    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _dropout_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, x_keep, output, scale, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
