import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _cross_entropy_kernel(logits, targets, output, TILE: ConstInt, ROWS: ConstInt):
    bid = ct.bid(0)
    cols = ct.arange(TILE, dtype=np.int32)

    for r in range(0, ROWS):
        row = bid * ROWS + r

        vals = ct.load(
            logits,
            index=(row, 0),
            shape=(1, TILE),
            latency=1,
            allow_tma=False,
        )
        vals_f = ct.astype(vals, np.float32).reshape((TILE,))

        max_val = ct.max(vals_f)

        shifted = (vals_f - max_val) * 1.4426950408889634
        exp_vals = ct.exp2(shifted, flush_to_zero=True)
        sum_exp = ct.sum(exp_vals)

        target = ct.astype(ct.load(targets, index=(row,), shape=()), np.int32)
        target_logit = ct.sum(ct.where(cols == target, vals_f, 0.0))

        loss = (max_val - target_logit) + ct.log2(sum_exp) * 0.6931471805599453
        ct.store(
            output,
            index=(row,),
            tile=ct.astype(loss, output.dtype),
            allow_tma=False,
        )


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    ROWS = 2
    occupancy = 8

    grid = (batch_size // ROWS, 1, 1)
    ct.launch(stream, grid, _cross_entropy_kernel, (logits, targets, output, TILE, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "ROWS": ROWS,
            "occupancy": occupancy,
            "serial_rows": True,
            "flat_tile": True,
            "target_from_tile": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
