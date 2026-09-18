import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _bitonic_step_desc(vals, SIZE: ConstInt, STRIDE: ConstInt, TILE: ConstInt):
    GROUPS = TILE // (2 * STRIDE)

    x = ct.reshape(vals, (GROUPS, 2, STRIDE))
    lo = ct.extract(x, (0, 0, 0), shape=(GROUPS, 1, STRIDE))
    hi = ct.extract(x, (0, 1, 0), shape=(GROUPS, 1, STRIDE))

    v_max = ct.maximum(lo, hi)
    v_min = ct.minimum(lo, hi)

    gids = ct.arange(GROUPS, dtype=np.int32)[:, None, None]
    descending_segment = ((gids * (2 * STRIDE)) & SIZE) == 0

    new_lo = ct.where(descending_segment, v_max, v_min)
    new_hi = ct.where(descending_segment, v_min, v_max)

    y = ct.cat((new_lo, new_hi), axis=1)
    return ct.reshape(y, (TILE,))


@ct.function
def _bitonic_sort_desc(vals, TILE: ConstInt, LOG_TILE: ConstInt):
    for p in range(1, LOG_TILE + 1):
        size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)
            vals = _bitonic_step_desc(vals, size, stride, TILE)
    return vals


@ct.kernel
def _topk_stage_kernel(src, dst, K: ConstInt, TILE: ConstInt, LOG_TILE: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )
    vals = _bitonic_sort_desc(vals, TILE, LOG_TILE)

    top_vals = ct.extract(vals, (0,), shape=(K,))
    ct.store(dst, index=(bid,), tile=top_vals, allow_tma=False)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    TILE = 2048
    LOG_TILE = 11
    occupancy = 8

    stream = torch.cuda.current_stream()
    kernel = _topk_stage_kernel.with_hints(occupancy=occupancy)

    cur = x
    n_cur = N
    stage_launches = 0

    while True:
        num_blocks = ct.cdiv(n_cur, TILE)

        if num_blocks == 1:
            output = torch.empty((k,), device=x.device, dtype=torch.float32)
            ct.launch(
                stream,
                (1, 1, 1),
                kernel,
                (cur, output, k, TILE, LOG_TILE),
            )
            stage_launches += 1
            break

        tmp_len = num_blocks * k
        tmp = torch.empty((tmp_len,), device=x.device, dtype=torch.float32)
        ct.launch(
            stream,
            (num_blocks, 1, 1),
            kernel,
            (cur, tmp, k, TILE, LOG_TILE),
        )
        stage_launches += 1
        cur = tmp
        n_cur = tmp_len

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "LOG_TILE": LOG_TILE,
        "occupancy": occupancy,
        "k": k,
        "stage_launches": stage_launches,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
