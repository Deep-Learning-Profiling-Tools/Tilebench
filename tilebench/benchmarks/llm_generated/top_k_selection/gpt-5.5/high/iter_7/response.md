```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


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
def _select_topl_stage_kernel(in_ptr, out_ptr, n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              LOCAL_K: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_offs = pid * BLOCK_SIZE + offs
    mask = in_offs < n_elements

    vals = tl.load(in_ptr + in_offs, mask=mask, other=-float("inf"))

    top_offs = tl.arange(0, LOCAL_K)
    top_vals = tl.full((LOCAL_K,), -float("inf"), dtype=tl.float32)

    for j in tl.static_range(0, LOCAL_K):
        m = tl.max(vals, axis=0)
        top_vals = tl.where(top_offs == j, m, top_vals)
        vals = tl.where(vals == m, -float("inf"), vals)

    tl.store(out_ptr + pid * LOCAL_K + top_offs, top_vals)


@triton.jit
def _topk_stage_kernel(in_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr,
                       LOG_BLOCK: tl.constexpr,
                       K_CONST: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_offs = pid * BLOCK_SIZE + offs
    mask = in_offs < n_elements

    vals = tl.load(in_ptr + in_offs, mask=mask, other=-float("inf"))
    vals = _bitonic_sort_desc(vals, BLOCK_SIZE, LOG_BLOCK)

    tl.store(out_ptr + pid * K_CONST + offs, vals, mask=offs < K_CONST)


@triton.jit
def _merge_topk_kernel(in_ptr, out_ptr, num_lists,
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


def _run_merge_path_reuse(x: torch.Tensor, N: int, k: int,
                          BLOCK_SIZE: int, LOG_BLOCK: int,
                          stage_warps: int, merge_warps: int,
                          num_stages: int):
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
        return output, 1, 0, 2 * k, (2 * k).bit_length() - 1

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

        _merge_topk_kernel[(next_lists,)](
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

    return output, 1, merge_launches, MERGE_SIZE, LOG_MERGE


def _choose_local_params(N: int, k: int):
    if N >= 1048576:
        if k >= 1024:
            return 1024, 10, 8
        if k >= 256:
            return 1024, 10, 8
        return 1024, 10, 4

    if N >= 262144:
        if k >= 1024:
            return 1024, 10, 16
        if k >= 256:
            return 1024, 10, 8
        return 1024, 10, 4

    return None


def _run_local_candidate_path(x: torch.Tensor, N: int, k: int,
                              SELECT_BLOCK: int, LOG_SELECT: int,
                              FINAL_BLOCK: int, LOG_FINAL: int,
                              LOCAL_K: int,
                              select_warps: int, stage_warps: int,
                              merge_warps: int, num_stages: int):
    num_blocks = triton.cdiv(N, SELECT_BLOCK)
    candidate_len = num_blocks * LOCAL_K
    candidates = torch.empty((candidate_len,), device=x.device, dtype=torch.float32)

    _select_topl_stage_kernel[(num_blocks,)](
        x, candidates, N,
        BLOCK_SIZE=SELECT_BLOCK,
        LOCAL_K=LOCAL_K,
        num_warps=select_warps,
        num_stages=num_stages,
    )

    output, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path_reuse(
        candidates, candidate_len, k,
        FINAL_BLOCK, LOG_FINAL,
        stage_warps, merge_warps,
        num_stages,
    )

    return output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    EXACT_BLOCK = 2048
    LOG_EXACT = 11
    FINAL_BLOCK = 2048
    LOG_FINAL = 11

    select_warps = 4
    stage_warps = 8
    merge_warps = 8
    num_stages = 4

    local_params = _choose_local_params(N, k)

    if local_params is not None:
        SELECT_BLOCK, LOG_SELECT, LOCAL_K = local_params
        output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_local_candidate_path(
            x, N, k,
            SELECT_BLOCK, LOG_SELECT,
            FINAL_BLOCK, LOG_FINAL,
            LOCAL_K,
            select_warps, stage_warps, merge_warps,
            num_stages,
        )

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "SELECT_BLOCK": SELECT_BLOCK,
            "LOG_SELECT": LOG_SELECT,
            "FINAL_BLOCK": FINAL_BLOCK,
            "LOG_FINAL": LOG_FINAL,
            "LOCAL_K": LOCAL_K,
            "K_CONST": k,
            "MERGE_SIZE": MERGE_SIZE,
            "LOG_MERGE": LOG_MERGE,
            "select_warps": select_warps,
            "stage_warps": stage_warps,
            "merge_warps": merge_warps,
            "num_stages": num_stages,
            "candidate_len": candidate_len,
            "local_candidate_path": 1,
            "exact_merge_path": 0,
            "select_launches": 1,
            "final_stage_launches": final_stage_launches,
            "merge_launches": merge_launches,
        })
        return output

    output, stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path_reuse(
        x, N, k,
        EXACT_BLOCK, LOG_EXACT,
        stage_warps, merge_warps,
        num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": EXACT_BLOCK,
        "LOG_BLOCK": LOG_EXACT,
        "K_CONST": k,
        "MERGE_SIZE": MERGE_SIZE,
        "LOG_MERGE": LOG_MERGE,
        "stage_warps": stage_warps,
        "merge_warps": merge_warps,
        "num_stages": num_stages,
        "local_candidate_path": 0,
        "exact_merge_path": 1,
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
_CANDIDATE_DISABLED: dict = {"value": 0}
_MERGE_DISABLED: dict = {"value": 0}


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


@ct.function
def _bitonic_merge_desc(vals, SIZE: ConstInt, LOG_SIZE: ConstInt):
    for q in range(LOG_SIZE, 0, -1):
        stride = 1 << (q - 1)
        groups = SIZE // (2 * stride)

        x = ct.reshape(vals, (groups, 2, stride))
        lo = ct.extract(x, (0, 0, 0), shape=(groups, 1, stride))
        hi = ct.extract(x, (0, 1, 0), shape=(groups, 1, stride))

        v_max = ct.maximum(lo, hi)
        v_min = ct.minimum(lo, hi)

        y = ct.cat((v_max, v_min), 1)
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
def _topk_stage_kernel(src, dst,
                       TILE: ConstInt,
                       LOG_TILE: ConstInt,
                       K_CONST: ConstInt):
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

    ct.store(dst, index=(bid,), tile=top, allow_tma=False)


@ct.kernel(occupancy=8)
def _merge_topk_kernel(src, dst, num_lists,
                       K_CONST: ConstInt,
                       MERGE_SIZE: ConstInt,
                       LOG_MERGE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(MERGE_SIZE, dtype=np.int32)

    left = bid * 2
    right = left + 1

    is_left = offs < K_CONST
    left_pos = left * K_CONST + offs
    right_pos = right * K_CONST + (MERGE_SIZE - 1 - offs)
    pos = ct.where(is_left, left_pos, right_pos)

    vals = ct.gather(src, pos, padding_value=-np.inf, check_bounds=True, latency=1)
    valid = is_left | (right < num_lists)
    vals = ct.where(valid, vals, -np.inf)

    vals = _bitonic_merge_desc(vals, MERGE_SIZE, LOG_MERGE)
    top = ct.extract(vals, (0,), shape=(K_CONST,))

    ct.store(dst, index=(bid,), tile=top, allow_tma=False)


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


def _choose_local_params(N: int, k: int):
    if N >= 1048576:
        if k >= 1024:
            return 1024, 8
        if k >= 256:
            return 1024, 8
        return 1024, 4

    if N >= 262144:
        if k >= 1024:
            return 1024, 16
        if k >= 256:
            return 1024, 8
        return 1024, 4

    return None


def _run_merge_path_reuse(x: torch.Tensor, N: int, k: int,
                          TILE: int, LOG_TILE: int):
    stream = torch.cuda.current_stream()
    num_lists = (N + TILE - 1) // TILE
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    if num_lists == 1:
        ct.launch(
            stream,
            (1, 1, 1),
            _topk_stage_kernel,
            (x, output, TILE, LOG_TILE, k),
        )
        return output, 1, 0, 2 * k, (2 * k).bit_length() - 1

    tmp_a = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)
    tmp_b = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_lists, 1, 1),
        _topk_stage_kernel,
        (x, tmp_a, TILE, LOG_TILE, k),
    )

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    cur = tmp_a
    other = tmp_b
    cur_lists = num_lists
    merge_launches = 0

    while cur_lists > 1:
        next_lists = (cur_lists + 1) // 2
        dst = output if next_lists == 1 else other

        ct.launch(
            stream,
            (next_lists, 1, 1),
            _merge_topk_kernel,
            (cur, dst, cur_lists, k, MERGE_SIZE, LOG_MERGE),
        )

        if next_lists != 1:
            cur, other = other, cur
        cur_lists = next_lists
        merge_launches += 1

    return output, 1, merge_launches, MERGE_SIZE, LOG_MERGE


def _run_local_candidate_path(x: torch.Tensor, N: int, k: int,
                              SELECT_TILE: int, LOCAL_K: int,
                              FINAL_TILE: int, LOG_FINAL: int):
    stream = torch.cuda.current_stream()

    num_blocks = (N + SELECT_TILE - 1) // SELECT_TILE
    candidate_len = num_blocks * LOCAL_K
    candidates = torch.empty((candidate_len,), device=x.device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_blocks, 1, 1),
        _select_topl_stage_kernel,
        (x, candidates, SELECT_TILE, LOCAL_K),
    )

    output, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path_reuse(
        candidates, candidate_len, k, FINAL_TILE, LOG_FINAL
    )

    return output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE


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
    grid_sort = ((sort_len + TILE - 1) // TILE, 1, 1)

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

    grid_copy = ((k + TILE - 1) // TILE, 1, 1)
    ct.launch(stream, grid_copy, _copy_topk_kernel, (src, output, TILE))

    return output, TILE, sort_len, pass_count, local_passes, global_passes


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    occupancy = 8
    FINAL_TILE = 2048
    LOG_FINAL = 11

    local_params = _choose_local_params(N, k)

    if local_params is not None and _CANDIDATE_DISABLED["value"] == 0:
        try:
            SELECT_TILE, LOCAL_K = local_params
            output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_local_candidate_path(
                x, N, k, SELECT_TILE, LOCAL_K, FINAL_TILE, LOG_FINAL
            )

            _LAST_CFG.clear()
            _LAST_CFG.update({
                "SELECT_TILE": SELECT_TILE,
                "LOCAL_K": LOCAL_K,
                "FINAL_TILE": FINAL_TILE,
                "LOG_FINAL": LOG_FINAL,
                "K_CONST": k,
                "MERGE_SIZE": MERGE_SIZE,
                "LOG_MERGE": LOG_MERGE,
                "occupancy": occupancy,
                "candidate_len": candidate_len,
                "local_candidate_path": 1,
                "exact_merge_path": 0,
                "full_sort_fallback": 0,
                "select_launches": 1,
                "final_stage_launches": final_stage_launches,
                "merge_launches": merge_launches,
            })
            return output
        except Exception:
            _CANDIDATE_DISABLED["value"] = 1

    if _MERGE_DISABLED["value"] == 0:
        try:
            output, stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path_reuse(
                x, N, k, FINAL_TILE, LOG_FINAL
            )

            _LAST_CFG.clear()
            _LAST_CFG.update({
                "TILE": FINAL_TILE,
                "LOG_TILE": LOG_FINAL,
                "K_CONST": k,
                "MERGE_SIZE": MERGE_SIZE,
                "LOG_MERGE": LOG_MERGE,
                "occupancy": occupancy,
                "local_candidate_path": 0,
                "exact_merge_path": 1,
                "full_sort_fallback": 0,
                "stage_launches": stage_launches,
                "merge_launches": merge_launches,
            })
            return output
        except Exception:
            _MERGE_DISABLED["value"] = 1

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
        "exact_merge_path": 0,
        "full_sort_fallback": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
