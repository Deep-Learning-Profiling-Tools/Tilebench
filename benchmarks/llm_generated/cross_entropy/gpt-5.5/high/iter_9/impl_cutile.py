import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _cross_entropy_kernel(
    logits,
    targets,
    output,
    TILE: ConstInt,
    ROWS: ConstInt,
    NUM_CHUNKS: ConstInt,
):
    bid = ct.bid(0)

    for r in range(0, ROWS):
        row = bid * ROWS + r

        vals0 = ct.load(
            logits,
            index=(row, 0),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        vals0_f = ct.astype(vals0, np.float32)
        exp0 = ct.exp2(vals0_f * 1.4426950408889634, flush_to_zero=True)
        sum_exp = ct.sum(exp0)

        for c in range(1, NUM_CHUNKS):
            vals = ct.load(
                logits,
                index=(row, c),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
                allow_tma=False,
            )
            vals_f = ct.astype(vals, np.float32)
            exp_vals = ct.exp2(vals_f * 1.4426950408889634, flush_to_zero=True)
            sum_exp = sum_exp + ct.sum(exp_vals)

        target = ct.astype(
            ct.load(
                targets,
                index=(row,),
                shape=(),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            ),
            np.int32,
        )
        target_logit = ct.astype(
            ct.gather(
                logits,
                (row, target),
                padding_value=0.0,
                check_bounds=False,
                latency=1,
            ),
            np.float32,
        )

        loss = ct.log2(sum_exp) * 0.6931471805599453 - target_logit
        ct.store(
            output,
            index=(row,),
            tile=ct.astype(loss, output.dtype),
            allow_tma=False,
        )


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()

    TILE = 256
    ROWS = 4
    NUM_CHUNKS = ct.cdiv(num_classes, TILE)
    occupancy = 8

    grid = (ct.cdiv(batch_size, ROWS), 1, 1)
    kernel = _cross_entropy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, targets, output, TILE, ROWS, NUM_CHUNKS))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "ROWS": ROWS,
            "NUM_CHUNKS": NUM_CHUNKS,
            "occupancy": occupancy,
            "serial_rows": True,
            "direct_logsumexp": True,
            "chunked_logsumexp": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
