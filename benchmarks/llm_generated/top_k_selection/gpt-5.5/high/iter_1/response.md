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
def _topk_stage_kernel(in_ptr, out_ptr, n_elements, K,
                       BLOCK_SIZE: tl.constexpr, LOG_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_offs = pid * BLOCK_SIZE + offs
    mask = in_offs < n_elements

    vals = tl.load(
        in_ptr + in_offs,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    vals = _bitonic_sort_desc(vals, BLOCK_SIZE, LOG_BLOCK)

    store_mask = offs < K
    tl.store(out_ptr + pid * K + offs, vals, mask=store_mask)


@triton.jit
def _merge_topk_kernel(in_ptr, out_ptr, num_lists, K,
                       MERGE_SIZE: tl.constexpr, LOG_MERGE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, MERGE_SIZE)

    left = pid * 2
    right = left + 1

    a_vals = tl.load(
        in_ptr + left * K + offs,
        mask=offs < K,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    b_pos = (MERGE_SIZE - 1) - offs
    b_vals = tl.load(
        in_ptr + right * K + b_pos,
        mask=(offs >= K) & (right < num_lists),
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    vals = tl.where(offs < K, a_vals, b_vals)
    vals = _bitonic_merge_desc(vals, MERGE_SIZE, LOG_MERGE)

    tl.store(out_ptr + pid * K + offs, vals, mask=offs < K)


def _run_iterative_fullsort(x: torch.Tensor, N: int, k: int,
                            BLOCK_SIZE: int, LOG_BLOCK: int,
                            num_warps: int, num_stages: int):
    cur = x
    n_cur = N
    stage_launches = 0

    while True:
        num_blocks = triton.cdiv(n_cur, BLOCK_SIZE)

        if num_blocks == 1:
            output = torch.empty((k,), device=x.device, dtype=torch.float32)
            _topk_stage_kernel[(1,)](
                cur, output, n_cur, k,
                BLOCK_SIZE=BLOCK_SIZE,
                LOG_BLOCK=LOG_BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            stage_launches += 1
            break

        tmp_len = num_blocks * k
        tmp = torch.empty((tmp_len,), device=x.device, dtype=torch.float32)
        _topk_stage_kernel[(num_blocks,)](
            cur, tmp, n_cur, k,
            BLOCK_SIZE=BLOCK_SIZE,
            LOG_BLOCK=LOG_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        stage_launches += 1
        cur = tmp
        n_cur = tmp_len

    return output, stage_launches


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    BLOCK_SIZE = 2048
    LOG_BLOCK = 11
    num_warps = 8
    num_stages = 4

    use_merge_path = (k >= 256) and (k <= 1024) and ((k & (k - 1)) == 0)

    if not use_merge_path:
        output, stage_launches = _run_iterative_fullsort(
            x, N, k, BLOCK_SIZE, LOG_BLOCK, num_warps, num_stages
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_SIZE": BLOCK_SIZE,
            "LOG_BLOCK": LOG_BLOCK,
            "MERGE_SIZE": 0,
            "LOG_MERGE": 0,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "k": k,
            "merge_path": 0,
            "stage_launches": stage_launches,
        })
        return output

    num_lists = triton.cdiv(N, BLOCK_SIZE)
    tmp = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    _topk_stage_kernel[(num_lists,)](
        x, tmp, N, k,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_BLOCK=LOG_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    stage_launches = 1
    merge_launches = 0

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    cur = tmp
    cur_lists = num_lists
    output = None

    while cur_lists > 1:
        next_lists = triton.cdiv(cur_lists, 2)
        if next_lists == 1:
            dst = torch.empty((k,), device=x.device, dtype=torch.float32)
            output = dst
        else:
            dst = torch.empty((next_lists * k,), device=x.device, dtype=torch.float32)

        _merge_topk_kernel[(next_lists,)](
            cur, dst, cur_lists, k,
            MERGE_SIZE=MERGE_SIZE,
            LOG_MERGE=LOG_MERGE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        cur = dst
        cur_lists = next_lists
        stage_launches += 1
        merge_launches += 1

    if output is None:
        output = cur

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "LOG_BLOCK": LOG_BLOCK,
        "MERGE_SIZE": MERGE_SIZE,
        "LOG_MERGE": LOG_MERGE,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "k": k,
        "merge_path": 1,
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
def _bitonic_pass_kernel(src, dst, size, stride, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)
    idx = bid * TILE + offs

    val = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    partner = idx ^ stride
    other = ct.gather(
        src,
        partner,
        padding_value=-np.inf,
        check_bounds=True,
    )

    v_max = ct.maximum(val, other)
    v_min = ct.minimum(val, other)

    descending_segment = (idx & size) == 0
    lower_partner = (idx & stride) == 0
    take_max = descending_segment == lower_partner

    out = ct.where(take_max, v_max, v_min)
    ct.store(dst, index=(bid,), tile=out)


@ct.kernel(occupancy=8)
def _copy_topk_kernel(src, output, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(output, index=(bid,), tile=vals)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    TILE = 1024
    occupancy = 8

    sort_len = 1 << ((N - 1).bit_length())
    temp_a = torch.empty((sort_len,), device=x.device, dtype=torch.float32)
    temp_b = torch.empty((sort_len,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    src = x
    dst = temp_a
    first_pass = True
    pass_count = 0

    log_n = sort_len.bit_length() - 1
    grid_sort = (ct.cdiv(sort_len, TILE), 1, 1)

    for p in range(1, log_n + 1):
        size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)
            ct.launch(stream, grid_sort, _bitonic_pass_kernel, (src, dst, size, stride, TILE))
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
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
