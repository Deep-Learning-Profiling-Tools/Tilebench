import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _bitonic_sort_desc(vals, SIZE: ConstInt, LOG_SIZE: ConstInt):
    for p in range(1, LOG_SIZE + 1):
        sort_size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)
            groups = SIZE // (2 * stride)

            x = ct.reshape(vals, (groups, 2, stride))
            lo = ct.extract(x, (0, 0, 0), shape=(groups, 1, stride))
            hi = ct.extract(x, (0, 1, 0), shape=(groups, 1, stride))

            v_max = ct.maximum(lo, hi)
            v_min = ct.minimum(lo, hi)

            gids = ct.arange(groups, dtype=np.int32)[:, None, None]
            group_base = gids * (2 * stride)
            descending_segment = (group_base & sort_size) == 0

            new_lo = ct.where(descending_segment, v_max, v_min)
            new_hi = ct.where(descending_segment, v_min, v_max)

            y = ct.cat((new_lo, new_hi), 1)
            vals = ct.reshape(y, (SIZE,))

    return vals


@ct.kernel(occupancy=8)
def _select_topl_stage_kernel(src, dst, TILE: ConstInt, LOCAL_K: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    top_idx = ct.arange(LOCAL_K, dtype=np.int32)
    selected = ct.full((LOCAL_K,), -np.inf, dtype=np.float32)

    for j in range(0, LOCAL_K):
        m = ct.max(vals)
        selected = ct.where(top_idx == j, m, selected)
        vals = ct.where(vals == m, -np.inf, vals)

    ct.store(dst, index=(bid,), tile=selected, allow_tma=False)


@ct.kernel(occupancy=8)
def _topk_stage_sort_kernel(src, dst, TILE: ConstInt, LOG_TILE: ConstInt, K_CONST: ConstInt):
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
    top = ct.extract(vals, (0,), shape=(K_CONST,))
    top2 = ct.reshape(top, (1, K_CONST))

    ct.store(dst, index=(bid, 0), tile=top2, allow_tma=False)


@ct.kernel(occupancy=8)
def _merge_sort_kernel(src, dst,
                       K_CONST: ConstInt,
                       MERGE_SIZE: ConstInt,
                       LOG_MERGE: ConstInt):
    bid = ct.bid(0)
    left = bid * 2
    right = left + 1

    lvals2 = ct.load(
        src,
        index=(left, 0),
        shape=(1, K_CONST),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )
    rvals2 = ct.load(
        src,
        index=(right, 0),
        shape=(1, K_CONST),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    lvals = ct.reshape(lvals2, (K_CONST,))
    rvals = ct.reshape(rvals2, (K_CONST,))
    vals = ct.cat((lvals, rvals), 0)

    vals = _bitonic_sort_desc(vals, MERGE_SIZE, LOG_MERGE)
    top = ct.extract(vals, (0,), shape=(K_CONST,))
    top2 = ct.reshape(top, (1, K_CONST))

    ct.store(dst, index=(bid, 0), tile=top2, allow_tma=False)


def _choose_local_k(N: int, k: int) -> int | None:
    if N >= 1048576:
        return 16
    if N >= 262144:
        if k >= 1024:
            return 64
        if k >= 256:
            return 32
        return 16
    return None


def _run_merge_path(src: torch.Tensor, N: int, k: int,
                    TILE: int, LOG_TILE: int):
    stream = torch.cuda.current_stream()

    num_lists = (N + TILE - 1) // TILE
    tmp_a = torch.empty((num_lists, k), device=src.device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_lists, 1, 1),
        _topk_stage_sort_kernel,
        (src, tmp_a, TILE, LOG_TILE, k),
    )

    cur = tmp_a
    cur_lists = num_lists
    merge_launches = 0

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    while cur_lists > 1:
        next_lists = (cur_lists + 1) // 2
        dst = torch.empty((next_lists, k), device=src.device, dtype=torch.float32)

        ct.launch(
            stream,
            (next_lists, 1, 1),
            _merge_sort_kernel,
            (cur, dst, k, MERGE_SIZE, LOG_MERGE),
        )

        cur = dst
        cur_lists = next_lists
        merge_launches += 1

    return cur.view(k), 1, merge_launches, MERGE_SIZE, LOG_MERGE


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    TILE = 2048
    LOG_TILE = 11
    occupancy = 8

    LOCAL_K = _choose_local_k(N, k)

    if LOCAL_K is not None:
        stream = torch.cuda.current_stream()
        num_blocks = (N + TILE - 1) // TILE
        candidate_len = num_blocks * LOCAL_K
        candidates = torch.empty((candidate_len,), device=x.device, dtype=torch.float32)

        ct.launch(
            stream,
            (num_blocks, 1, 1),
            _select_topl_stage_kernel,
            (x, candidates, TILE, LOCAL_K),
        )

        output, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path(
            candidates, candidate_len, k, TILE, LOG_TILE
        )

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "LOG_TILE": LOG_TILE,
            "LOCAL_K": LOCAL_K,
            "K_CONST": k,
            "MERGE_SIZE": MERGE_SIZE,
            "LOG_MERGE": LOG_MERGE,
            "occupancy": occupancy,
            "candidate_len": candidate_len,
            "local_candidate_path": 1,
            "exact_merge_path": 0,
            "select_launches": 1,
            "final_stage_launches": final_stage_launches,
            "merge_launches": merge_launches,
        })
        return output

    output, stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path(
        x, N, k, TILE, LOG_TILE
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "LOG_TILE": LOG_TILE,
        "K_CONST": k,
        "MERGE_SIZE": MERGE_SIZE,
        "LOG_MERGE": LOG_MERGE,
        "occupancy": occupancy,
        "local_candidate_path": 0,
        "exact_merge_path": 1,
        "stage_launches": stage_launches,
        "merge_launches": merge_launches,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
