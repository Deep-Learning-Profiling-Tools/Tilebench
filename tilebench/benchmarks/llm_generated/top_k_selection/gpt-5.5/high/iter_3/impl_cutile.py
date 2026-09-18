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


@ct.function
def _bitonic_merge_step_desc(vals, STRIDE: ConstInt, MERGE_SIZE: ConstInt):
    GROUPS = MERGE_SIZE // (2 * STRIDE)

    x = ct.reshape(vals, (GROUPS, 2, STRIDE))
    lo = ct.extract(x, (0, 0, 0), shape=(GROUPS, 1, STRIDE))
    hi = ct.extract(x, (0, 1, 0), shape=(GROUPS, 1, STRIDE))

    new_lo = ct.maximum(lo, hi)
    new_hi = ct.minimum(lo, hi)

    y = ct.cat((new_lo, new_hi), axis=1)
    return ct.reshape(y, (MERGE_SIZE,))


@ct.function
def _bitonic_merge_desc(vals, MERGE_SIZE: ConstInt, LOG_MERGE: ConstInt):
    for q in range(LOG_MERGE, 0, -1):
        stride = 1 << (q - 1)
        vals = _bitonic_merge_step_desc(vals, stride, MERGE_SIZE)
    return vals


@ct.kernel(occupancy=8)
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


@ct.kernel(occupancy=8)
def _merge_topk_kernel(src, dst, num_lists, K: ConstInt,
                       MERGE_SIZE: ConstInt, LOG_MERGE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(MERGE_SIZE, dtype=np.int32)

    left = bid * 2
    right = left + 1

    is_left = offs < K
    left_pos = left * K + offs
    right_pos = right * K + (MERGE_SIZE - 1 - offs)
    pos = ct.where(is_left, left_pos, right_pos)

    vals = ct.gather(
        src,
        pos,
        padding_value=-np.inf,
        check_bounds=True,
        latency=1,
    )

    vals = _bitonic_merge_desc(vals, MERGE_SIZE, LOG_MERGE)
    top_vals = ct.extract(vals, (0,), shape=(K,))
    ct.store(dst, index=(bid,), tile=top_vals, allow_tma=False)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    TILE = 2048
    LOG_TILE = 11
    occupancy = 8

    stream = torch.cuda.current_stream()

    num_lists = ct.cdiv(N, TILE)
    tmp = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_lists, 1, 1),
        _topk_stage_kernel,
        (x, tmp, k, TILE, LOG_TILE),
    )

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    cur = tmp
    cur_lists = num_lists
    stage_launches = 1
    merge_launches = 0

    while cur_lists > 1:
        next_lists = ct.cdiv(cur_lists, 2)
        dst_len = k if next_lists == 1 else next_lists * k
        dst = torch.empty((dst_len,), device=x.device, dtype=torch.float32)

        ct.launch(
            stream,
            (next_lists, 1, 1),
            _merge_topk_kernel,
            (cur, dst, cur_lists, k, MERGE_SIZE, LOG_MERGE),
        )

        cur = dst
        cur_lists = next_lists
        stage_launches += 1
        merge_launches += 1

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "LOG_TILE": LOG_TILE,
        "MERGE_SIZE": MERGE_SIZE,
        "LOG_MERGE": LOG_MERGE,
        "occupancy": occupancy,
        "k": k,
        "stage_launches": stage_launches,
        "merge_launches": merge_launches,
        "merge_path": 1,
    })
    return cur


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
