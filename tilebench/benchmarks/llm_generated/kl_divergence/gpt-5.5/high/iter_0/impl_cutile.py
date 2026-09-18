import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _kl_divergence_kernel(log_y_pred, y_true, output, COLS: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    total = ct.full((1,), 0.0, dtype=np.float32)
    num_tiles = ct.cdiv(COLS, TILE)

    for j in range(0, num_tiles):
        if (j + 1) * TILE <= COLS:
            y = ct.astype(
                ct.load(
                    y_true,
                    index=(row, j),
                    shape=(1, TILE),
                    latency=1,
                    allow_tma=False,
                ),
                np.float32,
            )
            log_pred = ct.astype(
                ct.load(
                    log_y_pred,
                    index=(row, j),
                    shape=(1, TILE),
                    latency=1,
                    allow_tma=False,
                ),
                np.float32,
            )
        else:
            y = ct.astype(
                ct.load(
                    y_true,
                    index=(row, j),
                    shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1,
                    allow_tma=False,
                ),
                np.float32,
            )
            log_pred = ct.astype(
                ct.load(
                    log_y_pred,
                    index=(row, j),
                    shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1,
                    allow_tma=False,
                ),
                np.float32,
            )

        y_log_arg = ct.where(y > 0.0, y, 1.0)
        term = y * (ct.log(y_log_arg) - log_pred)
        total = total + ct.sum(term, axis=1, keepdims=False)

    ct.store(output, index=(row,), tile=total, allow_tma=False)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    log_y_pred = log_y_pred.contiguous()
    y_true = y_true.contiguous()

    rows = log_y_pred.shape[0]
    cols = log_y_pred.shape[1]
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    grid = (rows, 1, 1)
    kernel = _kl_divergence_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
