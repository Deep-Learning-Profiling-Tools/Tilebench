import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _ce_kernel(logits, targets, output,
               TILE_N: ConstInt, ROWS: ConstInt):
    bid = ct.bid(0)
    x = ct.load(logits, index=(bid, 0), shape=(ROWS, TILE_N),
                padding_mode=ct.PaddingMode.NEG_INF)
    x_f32 = ct.astype(x, np.float32)

    m = ct.max(x_f32, axis=1, keepdims=True)          # [ROWS, 1]
    z = ct.exp(x_f32 - m)
    s = ct.sum(z, axis=1, keepdims=False)             # [ROWS]
    log_s = ct.log(s)
    m_flat = ct.reshape(m, (ROWS,))

    target_tile = ct.load(targets, index=(bid,), shape=(ROWS,))
    target_i32 = ct.astype(target_tile, np.int32)
    row_idx = ct.arange(ROWS, dtype=np.int32) + bid * ROWS
    target_logit = ct.gather(logits, (row_idx, target_i32))
    target_f32 = ct.astype(target_logit, np.float32)

    loss = -(target_f32 - m_flat) + log_s
    out_tile = ct.astype(loss, logits.dtype)
    ct.store(output, index=(bid,), tile=out_tile)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
    stream = torch.cuda.current_stream()

    TILE_N = 1
    while TILE_N < num_classes:
        TILE_N <<= 1
    if TILE_N < 256:
        TILE_N = 256

    # Multi-row tile: pick ROWS that divides batch_size where possible.
    ROWS = 8
    while ROWS > 1 and (batch_size % ROWS) != 0:
        ROWS //= 2
    occupancy = 4

    grid = (batch_size // ROWS, 1, 1)
    ct.launch(stream, grid, _ce_kernel,
              (logits, targets, output, TILE_N, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "ROWS": ROWS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
