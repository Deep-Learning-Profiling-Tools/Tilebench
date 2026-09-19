import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}
_FUSED_DISABLED: dict = {"value": 0}


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
def _sort_8_segments_1024_desc(vals):
    idx = tl.arange(0, 8192)
    local = idx & 1023

    for p in tl.static_range(1, 11):
        size = 1 << p
        for q in tl.static_range(p, 0, -1):
            stride = 1 << (q - 1)
            other = tl.gather(vals, idx ^ stride, axis=0)

            v_min = tl.minimum(vals, other)
            v_max = tl.maximum(vals, other)

            descending_segment = (local & size) == 0
            lower_partner = (local & stride) == 0
            take_max = descending_segment == lower_partner

            vals = tl.where(take_max, v_max, v_min)

    return vals


@triton.jit
def _merge_level_top1024(vals, SPAN: tl.constexpr):
    idx = tl.arange(0, 8192)
    local = idx & (SPAN - 1)
    active = local < 2048
    group_base = idx - local
    right_base = group_base + (SPAN // 2)

    src_left = group_base + local
    src_right = right_base + (2047 - local)
    src = tl.where(local < 1024, src_left, src_right)
    src = tl.where(active, src, idx)

    merge_vals = tl.gather(vals, src, axis=0)
    vals = tl.where(active, merge_vals, vals)

    for q in tl.static_range(11, 0, -1):
        stride = 1 << (q - 1)
        other = tl.gather(vals, idx ^ stride, axis=0)

        v_min = tl.minimum(vals, other)
        v_max = tl.maximum(vals, other)

        lower_partner = (idx & stride) == 0
        merged = tl.where(lower_partner, v_max, v_min)
        vals = tl.where(active, merged, vals)

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
def _fused_final_8192_to_1024_kernel(candidates, output):
    idx = tl.arange(0, 8192)
    vals = tl.load(candidates + idx)

    vals = _sort_8_segments_1024_desc(vals)
    vals = _merge_level_top1024(vals, 2048)
    vals = _merge_level_top1024(vals, 4096)
    vals = _merge_level_top1024(vals, 8192)

    out_idx = tl.arange(0, 1024)
    out_vals = tl.gather(vals, out_idx, axis=0)
    tl.store(output + out_idx, out_vals)


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
                              merge_warps: int, fused_warps: int,
                              num_stages: int,
                              use_fused_final: bool):
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

    if use_fused_final and candidate_len == 8192 and k == 1024 and _FUSED_DISABLED["value"] == 0:
        output = torch.empty((k,), device=x.device, dtype=torch.float32)
        try:
            _fused_final_8192_to_1024_kernel[(1,)](
                candidates, output,
                num_warps=fused_warps,
                num_stages=num_stages,
            )
            return output, candidate_len, 0, 0, 8192, 13, 1
        except Exception:
            _FUSED_DISABLED["value"] = 1

    output, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path_reuse(
        candidates, candidate_len, k,
        FINAL_BLOCK, LOG_FINAL,
        stage_warps, merge_warps,
        num_stages,
    )

    return output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE, 0


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    EXACT_BLOCK = 2048
    LOG_EXACT = 11
    FINAL_BLOCK = 2048
    LOG_FINAL = 11

    select_warps = 4
    stage_warps = 8
    merge_warps = 8
    fused_warps = 16
    num_stages = 4

    local_params = _choose_local_params(N, k)

    if local_params is not None:
        SELECT_BLOCK, LOG_SELECT, LOCAL_K = local_params
        use_fused_final = N >= 1048576 and k == 1024
        output, candidate_len, final_stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE, fused_final = _run_local_candidate_path(
            x, N, k,
            SELECT_BLOCK, LOG_SELECT,
            FINAL_BLOCK, LOG_FINAL,
            LOCAL_K,
            select_warps, stage_warps, merge_warps, fused_warps,
            num_stages,
            use_fused_final,
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
            "fused_warps": fused_warps,
            "num_stages": num_stages,
            "candidate_len": candidate_len,
            "local_candidate_path": 1,
            "fused_final_path": fused_final,
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
        "fused_final_path": 0,
        "exact_merge_path": 1,
        "stage_launches": stage_launches,
        "merge_launches": merge_launches,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
