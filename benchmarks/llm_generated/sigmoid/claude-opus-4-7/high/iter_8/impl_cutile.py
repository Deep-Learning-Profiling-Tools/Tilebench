import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

_LOG2E = 1.4426950408889634


@ct.kernel(occupancy=2)
def _sigmoid_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2(e)))
    neg_scaled = xf * (-_LOG2E)
    e = ct.exp2(neg_scaled, flush_to_zero=True)
    y = 1.0 / (1.0 + e)
    y_out = ct.astype(y, x.dtype)
    ct.store(output, index=(bid,), tile=y_out)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 16384
    occupancy = 2

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _sigmoid_kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy,
                      "math": "exp2_approx"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
