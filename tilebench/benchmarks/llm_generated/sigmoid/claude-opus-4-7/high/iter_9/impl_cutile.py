import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _sigmoid_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO,
                     allow_tma=False)
    xf = ct.astype(x_tile, np.float32)
    # sigmoid(x) = 0.5 + 0.5 * tanh(x / 2)
    # ct.tanh maps to tanh.approx.f32 on Blackwell — single PTX instr,
    # cheaper than exp2 + reciprocal + add path.
    y = 0.5 + 0.5 * ct.tanh(xf * 0.5)
    y_out = ct.astype(y, x.dtype)
    ct.store(output, index=(bid,), tile=y_out, allow_tma=False)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _sigmoid_kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy,
                      "math": "tanh_approx", "allow_tma": False})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
