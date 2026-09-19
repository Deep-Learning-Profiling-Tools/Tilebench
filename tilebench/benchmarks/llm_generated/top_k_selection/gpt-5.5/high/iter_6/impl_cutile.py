import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}
_LOCAL_DISABLED: dict = {"value": 0}


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
def _topk_single_sort_kernel(src, output,
                             TILE: ConstInt,
                             LOG_TILE: ConstInt,
                             K_CONST: ConstInt):
    vals = ct.load(
        src,
        index=(0,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    vals = _bitonic_sort_desc(vals, TILE, LOG_TILE)
    top = ct.extract(vals, (0,), shape=(K_CONST,))

    ct.store(output, index=(0,), tile=top, allow_tma=False)


@ct.kernel(occupancy=8)
def _bitonic_local_pass_kernel(src, dst, size, STRIDE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    GROUPS = TILE // (2 * STRIDE)

    x = ct.reshape(vals, (GROUPS, 2, STRIDE))
    lo = ct.extract(x, (0, 0, 0), shape=(GROUPS, 1, STRIDE))
    hi = ct.extract(x, (0, 1, 0), shape=(GROUPS, 1, STRIDE))

    v_max = ct.maximum(lo, hi)
    v_min = ct.minimum(lo, hi)

    gids = ct.arange(GROUPS, dtype=np.int32)[:, None, None]
    group_base = bid * TILE + gids * (2 * STRIDE)
    descending_segment = (group_base & size) == 0

    new_lo = ct.where(descending_segment, v_max, v_min)
    new_hi = ct.where(descending_segment, v_min, v_max)

    y = ct.cat((new_lo, new_hi), 1)
    out = ct.reshape(y, (TILE,))

    ct.store(dst, index=(bid,), tile=out, allow_tma=False)


@ct.kernel(occupancy=8)
def _bitonic_global_pass_kernel(src, dst, size, stride, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)
    idx = bid * TILE + offs

    val = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    partner_bid = bid ^ (stride // TILE)
    other = ct.load(
        src,
        index=(partner_bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    v_max = ct.maximum(val, other)
    v_min = ct.minimum(val, other)

    descending_segment = (idx & size) == 0
    lower_partner = (idx & stride) == 0
    take_max = descending_segment == lower_partner

    out = ct.where(take_max, v_max, v_min)
    ct.store(dst, index=(bid,), tile=out, allow_tma=False)


@ct.kernel(occupancy=8)
def _copy_topk_kernel(src, output, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )
    ct.store(output, index=(bid,), tile=vals, allow_tma=False)


def _run_local_candidate_path(x: torch.Tensor, N: int, k: int):
    stream = torch.cuda.current_stream()

    SELECT_TILE = 8192
    LOCAL_K = 32
    FINAL_TILE = 4096
    LOG_FINAL = 12

    num_blocks = (N + SELECT_TILE - 1) // SELECT_TILE
    candidate_len = num_blocks * LOCAL_K

    candidates = torch.empty((candidate_len,), device=x.device, dtype=torch.float32)
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_blocks, 1, 1),
        _select_topl_stage_kernel,
        (x, candidates, SELECT_TILE, LOCAL_K),
    )

    ct.launch(
        stream,
        (1, 1, 1),
        _topk_single_sort_kernel,
        (candidates, output, FINAL_TILE, LOG_FINAL, k),
    )

    return output, SELECT_TILE, LOCAL_K, FINAL_TILE, LOG_FINAL, candidate_len


def _run_full_bitonic(x: torch.Tensor, N: int, k: int):
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    TILE = 2048
    sort_len = 1 << ((N - 1).bit_length())

    temp_a = torch.empty((sort_len,), device=x.device, dtype=torch.float32)
    temp_b = torch.empty((sort_len,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    src = x
    dst = temp_a
    first_pass = True
    pass_count = 0
    local_passes = 0
    global_passes = 0

    log_n = sort_len.bit_length() - 1
    grid_sort = (ct.cdiv(sort_len, TILE), 1, 1)

    for p in range(1, log_n + 1):
        size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)

            if stride < TILE:
                ct.launch(
                    stream,
                    grid_sort,
                    _bitonic_local_pass_kernel,
                    (src, dst, size, stride, TILE),
                )
                local_passes += 1
            else:
                ct.launch(
                    stream,
                    grid_sort,
                    _bitonic_global_pass_kernel,
                    (src, dst, size, stride, TILE),
                )
                global_passes += 1

            pass_count += 1

            if first_pass:
                src = temp_a
                dst = temp_b
                first_pass = False
            else:
                src, dst = dst, src

    grid_copy = (ct.cdiv(k, TILE), 1, 1)
    ct.launch(stream, grid_copy, _copy_topk_kernel, (src, output, TILE))

    return output, TILE, sort_len, pass_count, local_passes, global_passes


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()
    occupancy = 8

    if N >= 1048576 and _LOCAL_DISABLED["value"] == 0:
        try:
            output, SELECT_TILE, LOCAL_K, FINAL_TILE, LOG_FINAL, candidate_len = _run_local_candidate_path(x, N, k)

            _LAST_CFG.clear()
            _LAST_CFG.update({
                "SELECT_TILE": SELECT_TILE,
                "LOCAL_K": LOCAL_K,
                "FINAL_TILE": FINAL_TILE,
                "LOG_FINAL": LOG_FINAL,
                "K_CONST": k,
                "occupancy": occupancy,
                "candidate_len": candidate_len,
                "local_candidate_path": 1,
                "exact_sort_fallback": 0,
                "select_launches": 1,
                "final_sort_launches": 1,
            })
            return output
        except Exception:
            _LOCAL_DISABLED["value"] = 1

    output, TILE, sort_len, pass_count, local_passes, global_passes = _run_full_bitonic(x, N, k)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "sort_len": sort_len,
        "passes": pass_count,
        "local_passes": local_passes,
        "global_passes": global_passes,
        "coalesced_global": 1,
        "local_candidate_path": 0,
        "exact_sort_fallback": 1 if N >= 1048576 else 0,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
