import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _kl_kernel(log_y_pred, y_true, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    one = ct.full((1, TILE), 1.0, dtype=np.float32)
    for j in range(0, num_tiles):
        lp = ct.load(log_y_pred, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        yt = ct.load(y_true, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        # When yt == 0, term = 0 * anything = 0, so we only need to make
        # the log argument safe (avoid log(0) = -inf which would NaN-poison).
        safe = ct.where(yt > 0.0, yt, one)
        term = yt * (ct.log(safe) - lp)
        acc = acc + term
    total = ct.sum(acc, axis=1, keepdims=False)
    ct.store(output, index=(row,), tile=total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8
    grid = (rows, 1, 1)
    kernel = _kl_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
