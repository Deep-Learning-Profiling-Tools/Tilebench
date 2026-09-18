```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}
_STATE: dict = {"disable_rank_merge": False}


@triton.jit
def _bitonic_sort_desc(vals, BLOCK_SIZE: tl.constexpr, LOG_BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)

    for p in tl.static_range(1, LOG_BLOCK + 1):
        size = 1 << p
        for q in tl.static_range(p, 0, -1):
            stride = 1 << (q - 1)
            other = tl.gather(vals, idx ^ stride, axis=0)

            v_min = tl.minimum(vals, other)
            v_max = tl.maximum(vals, other)

            descending_segment = (idx & size) == 0
            lower_partner = (idx & stride) == 0
            take_max = descending_segment == lower_partner

            vals = tl.where(take_max, v_max, v_min)

    return vals


@triton.jit
def _bitonic_merge_desc(vals, MERGE_SIZE: tl.constexpr, LOG_MERGE: tl.constexpr):
    idx = tl.arange(0, MERGE_SIZE)

    for q in tl.static_range(LOG_MERGE, 0, -1):
        stride = 1 << (q - 1)
        other = tl.gather(vals, idx ^ stride, axis=0)

        v_min = tl.minimum(vals, other)
        v_max = tl.maximum(vals, other)

        lower_partner = (idx & stride) == 0
        vals = tl.where(lower_partner, v_max, v_min)

    return vals


@triton.jit
def _topk_stage_kernel(in_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr, LOG_BLOCK: tl.constexpr,
                       K_CONST: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_offs = pid * BLOCK_SIZE + offs
    mask = in_offs < n_elements

    vals = tl.load(in_ptr + in_offs, mask=mask, other=-float("inf"))
    vals = _bitonic_sort_desc(vals, BLOCK_SIZE, LOG_BLOCK)

    tl.store(out_ptr + pid * K_CONST + offs, vals, mask=offs < K_CONST)


@triton.jit
def _merge_rank_kernel(in_ptr, out_ptr, num_lists,
                       K_CONST: tl.constexpr, LOG_K: tl.constexpr):
    pid = tl.program_id(0)
    ranks = tl.arange(0, K_CONST)

    left = pid * 2
    right = left + 1

    lvals = tl.load(
        in_ptr + left * K_CONST + ranks,
        mask=left < num_lists,
        other=-float("inf"),
    )
    rvals = tl.load(
        in_ptr + right * K_CONST + ranks,
        mask=right < num_lists,
        other=-float("inf"),
    )

    lo = tl.zeros((K_CONST,), dtype=tl.int32)
    hi = ranks

    for _ in tl.static_range(0, LOG_K):
        active = lo < hi
        mid = (lo + hi) // 2
        j = ranks - mid

        l_mid = tl.gather(lvals, mid, axis=0)
        r_prev_idx = tl.maximum(j - 1, 0)
        r_prev = tl.gather(rvals, r_prev_idx, axis=0)

        go_right = (j > 0) & (l_mid > r_prev)
        cond = active & go_right

        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(active & (~go_right), mid, hi)

    i = lo
    j = ranks - i

    l_i = tl.gather(lvals, i, axis=0)
    r_j = tl.gather(rvals, j, axis=0)
    out = tl.maximum(l_i, r_j)

    tl.store(out_ptr + pid * K_CONST + ranks, out)


@triton.jit
def _merge_bitonic_kernel(in_ptr, out_ptr, num_lists,
                          K_CONST: tl.constexpr,
                          MERGE_SIZE: tl.constexpr,
                          LOG_MERGE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, MERGE_SIZE)

    left = pid * 2
    right = left + 1

    is_left_half = offs < K_CONST
    left_pos = left * K_CONST + offs
    right_pos = right * K_CONST + (MERGE_SIZE - 1 - offs)
    pos = tl.where(is_left_half, left_pos, right_pos)

    vals = tl.load(
        in_ptr + pos,
        mask=is_left_half | (right < num_lists),
        other=-float("inf"),
    )

    vals = _bitonic_merge_desc(vals, MERGE_SIZE, LOG_MERGE)
    tl.store(out_ptr + pid * K_CONST + offs, vals, mask=offs < K_CONST)


def _run_rank_merge_path(x: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 1024
    LOG_BLOCK = 10
    stage_warps = 8
    merge_warps = 8
    num_stages = 4
    LOG_K = k.bit_length() - 1

    num_lists = triton.cdiv(N, BLOCK_SIZE)
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    if num_lists == 1:
        _topk_stage_kernel[(1,)](
            x, output, N,
            BLOCK_SIZE=BLOCK_SIZE,
            LOG_BLOCK=LOG_BLOCK,
            K_CONST=k,
            num_warps=stage_warps,
            num_stages=num_stages,
        )
        return output, 1, 0, BLOCK_SIZE, LOG_BLOCK, LOG_K

    tmp_a = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)
    tmp_b = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    _topk_stage_kernel[(num_lists,)](
        x, tmp_a, N,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_BLOCK=LOG_BLOCK,
        K_CONST=k,
        num_warps=stage_warps,
        num_stages=num_stages,
    )

    cur = tmp_a
    other = tmp_b
    cur_lists = num_lists
    merge_launches = 0

    while cur_lists > 1:
        next_lists = triton.cdiv(cur_lists, 2)
        dst = output if next_lists == 1 else other

        _merge_rank_kernel[(next_lists,)](
            cur, dst, cur_lists,
            K_CONST=k,
            LOG_K=LOG_K,
            num_warps=merge_warps,
            num_stages=num_stages,
        )

        if next_lists != 1:
            cur, other = other, cur
        cur_lists = next_lists
        merge_launches += 1

    return output, 1, merge_launches, BLOCK_SIZE, LOG_BLOCK, LOG_K


def _run_bitonic_fallback(x: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 2048
    LOG_BLOCK = 11
    stage_warps = 8
    merge_warps = 8
    num_stages = 4

    num_lists = triton.cdiv(N, BLOCK_SIZE)
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    if num_lists == 1:
        _topk_stage_kernel[(1,)](
            x, output, N,
            BLOCK_SIZE=BLOCK_SIZE,
            LOG_BLOCK=LOG_BLOCK,
            K_CONST=k,
            num_warps=stage_warps,
            num_stages=num_stages,
        )
        return output, 1, 0, BLOCK_SIZE, LOG_BLOCK, 0, 0

    tmp_a = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)
    tmp_b = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    _topk_stage_kernel[(num_lists,)](
        x, tmp_a, N,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_BLOCK=LOG_BLOCK,
        K_CONST=k,
        num_warps=stage_warps,
        num_stages=num_stages,
    )

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    cur = tmp_a
    other = tmp_b
    cur_lists = num_lists
    merge_launches = 0

    while cur_lists > 1:
        next_lists = triton.cdiv(cur_lists, 2)
        dst = output if next_lists == 1 else other

        _merge_bitonic_kernel[(next_lists,)](
            cur, dst, cur_lists,
            K_CONST=k,
            MERGE_SIZE=MERGE_SIZE,
            LOG_MERGE=LOG_MERGE,
            num_warps=merge_warps,
            num_stages=num_stages,
        )

        if next_lists != 1:
            cur, other = other, cur
        cur_lists = next_lists
        merge_launches += 1

    return output, 1, merge_launches, BLOCK_SIZE, LOG_BLOCK, MERGE_SIZE, LOG_MERGE


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    stage_warps = 8
    merge_warps = 8
    num_stages = 4

    use_rank_merge = (
        (not _STATE.get("disable_rank_merge", False))
        and (k >= 1)
        and ((k & (k - 1)) == 0)
        and (k <= 1024)
    )

    if use_rank_merge:
        try:
            output, stage_launches, merge_launches, BLOCK_SIZE, LOG_BLOCK, LOG_K = _run_rank_merge_path(x, N, k)
            _LAST_CFG.clear()
            _LAST_CFG.update({
                "BLOCK_SIZE": BLOCK_SIZE,
                "LOG_BLOCK": LOG_BLOCK,
                "K_CONST": k,
                "LOG_K": LOG_K,
                "stage_warps": stage_warps,
                "merge_warps": merge_warps,
                "num_stages": num_stages,
                "rank_merge": 1,
                "bitonic_fallback": 0,
                "stage_launches": stage_launches,
                "merge_launches": merge_launches,
            })
            return output
        except Exception:
            _STATE["disable_rank_merge"] = True

    output, stage_launches, merge_launches, BLOCK_SIZE, LOG_BLOCK, MERGE_SIZE, LOG_MERGE = _run_bitonic_fallback(x, N, k)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "LOG_BLOCK": LOG_BLOCK,
        "K_CONST": k,
        "MERGE_SIZE": MERGE_SIZE,
        "LOG_MERGE": LOG_MERGE,
        "stage_warps": stage_warps,
        "merge_warps": merge_warps,
        "num_stages": num_stages,
        "rank_merge": 0,
        "bitonic_fallback": 1,
        "stage_launches": stage_launches,
        "merge_launches": merge_launches,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


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


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    TILE = 2048
    occupancy = 8

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

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "sort_len": sort_len,
        "passes": pass_count,
        "local_passes": local_passes,
        "global_passes": global_passes,
        "coalesced_global": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
