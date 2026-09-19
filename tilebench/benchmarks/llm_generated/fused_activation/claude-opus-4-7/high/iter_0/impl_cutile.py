import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _fused_act_kernel(x, gate, bias, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_t = ct.load(x, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    g_t = ct.load(gate, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    b_t = ct.load(bias, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    z = x_t * g_t + b_t
    y = z / (1.0 + ct.exp(-z))
    ct.store(output, index=(bid,), tile=y)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n, TILE), 1, 1)
    kernel = _fused_act_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, gate, bias, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
