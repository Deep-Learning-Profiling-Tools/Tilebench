import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _cross_entropy_kernel(logits, targets, output, TILE: ConstInt):
    row = ct.bid(0)

    vals = ct.load(
        logits,
        index=(row, 0),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    vals_f = ct.astype(vals, np.float32)

    max_val = ct.max(vals_f)

    shifted = (vals_f - max_val) * 1.4426950408889634
    exp_vals = ct.exp2(shifted, flush_to_zero=True)
    sum_exp = ct.sum(exp_vals)

    target = ct.astype(ct.load(targets, index=(row,), shape=()), np.int32)
    target_logit = ct.astype(
        ct.gather(logits, (row, target), padding_value=0.0, latency=1),
        np.float32,
    )

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
    occupancy = 8

    grid = (batch_size, 1, 1)
    kernel = _cross_entropy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, targets, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
