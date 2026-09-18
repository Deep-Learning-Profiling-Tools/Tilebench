import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _kl_kernel(log_y_pred, y_true, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        lp = ct.load(log_y_pred, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        yt = ct.load(y_true, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        pos = yt > 0.0
        safe = ct.where(pos, yt, ct.full((1, TILE), 1.0, dtype=np.float32))
        log_yt = ct.log(safe)
        term = ct.where(pos, yt * (log_yt - lp),
                        ct.full((1, TILE), 0.0, dtype=np.float32))
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
    grid = (rows, 1, 1)
    ct.launch(stream, grid, _kl_kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 8})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
