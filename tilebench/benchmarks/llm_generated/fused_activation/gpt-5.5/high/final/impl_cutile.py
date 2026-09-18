import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _fused_activation_kernel(x, gate, bias, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.astype(
        ct.load(
            x,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        ),
        np.float32,
    )
    gate_tile = ct.astype(
        ct.load(
            gate,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        ),
        np.float32,
    )
    bias_tile = ct.astype(
        ct.load(
            bias,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        ),
        np.float32,
    )

    z = x_tile * gate_tile + bias_tile
    sigmoid = 1.0 / (1.0 + ct.exp2(-z * 1.4426950408889634, flush_to_zero=True))
    y = z * sigmoid

    ct.store(output, index=(bid,), tile=ct.astype(y, output.dtype), allow_tma=False)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _fused_activation_kernel, (x, gate, bias, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
