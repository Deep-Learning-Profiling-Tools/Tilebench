import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _ce_kernel(logits, targets, output,
               NUM_CLASSES: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)
    x = ct.load(logits, index=(row, 0), shape=(1, TILE_N),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1)
    x_f32 = ct.astype(x, np.float32)

    m = ct.max(x_f32)
    z = ct.exp(x_f32 - m)
    s = ct.sum(z)
    log_s = ct.log(s)

    # Load target index
    target_tile = ct.load(targets, index=(row,), shape=(1,))
    target_i32 = ct.astype(target_tile, np.int32)

    # Extract target logit via masked reduction from already-loaded tile
    cols = ct.arange(TILE_N, dtype=np.int32)
    cols_2d = ct.reshape(cols, (1, TILE_N))
    target_2d = ct.reshape(target_i32, (1, 1))
    mask = cols_2d == target_2d
    target_logit = ct.sum(ct.where(mask, x_f32, 0.0))

    loss = -(target_logit - m) + log_s
    out_tile = ct.astype(loss, logits.dtype)
    ct.store(output, index=(row,), tile=out_tile)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
    stream = torch.cuda.current_stream()

    TILE_N = 1
    while TILE_N < num_classes:
        TILE_N <<= 1
    if TILE_N < 256:
        TILE_N = 256

    grid = (batch_size, 1, 1)
    ct.launch(stream, grid, _ce_kernel,
              (logits, targets, output, num_classes, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": 8,
                      "latency": 1,
                      "opt": "target_via_masked_reduce"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
