```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}
_STATE: dict = {"disable_merge": False}


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

    tl.store(out_ptr + pid * K + offs, vals, mask=offs < K)


@triton.jit
def _merge_topk_kernel(in_ptr, out_ptr, num_lists, K,
                       MERGE_SIZE: tl.constexpr, LOG_MERGE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, MERGE_SIZE)

    left = pid * 2
    right = left + 1

    is_left_half = offs < K
    left_pos = left * K + offs
    right_pos = right * K + (MERGE_SIZE - 1 - offs)
    pos = tl.where(is_left_half, left_pos, right_pos)

    load_mask = is_left_half | ((offs >= K) & (right < num_lists))
    vals = tl.load(
        in_ptr + pos,
        mask=load_mask,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

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


def _run_merge_path(x: torch.Tensor, N: int, k: int,
                    BLOCK_SIZE: int, LOG_BLOCK: int,
                    stage_warps: int, merge_warps: int,
                    num_stages: int):
    num_lists = triton.cdiv(N, BLOCK_SIZE)
    tmp = torch.empty((num_lists * k,), device=x.device, dtype=torch.float32)

    _topk_stage_kernel[(num_lists,)](
        x, tmp, N, k,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_BLOCK=LOG_BLOCK,
        num_warps=stage_warps,
        num_stages=num_stages,
    )

    stage_launches = 1
    merge_launches = 0

    MERGE_SIZE = 2 * k
    LOG_MERGE = MERGE_SIZE.bit_length() - 1

    cur = tmp
    cur_lists = num_lists

    while cur_lists > 1:
        next_lists = triton.cdiv(cur_lists, 2)
        dst_len = k if next_lists == 1 else next_lists * k
        dst = torch.empty((dst_len,), device=x.device, dtype=torch.float32)

        _merge_topk_kernel[(next_lists,)](
            cur, dst, cur_lists, k,
            MERGE_SIZE=MERGE_SIZE,
            LOG_MERGE=LOG_MERGE,
            num_warps=merge_warps,
            num_stages=num_stages,
        )

        cur = dst
        cur_lists = next_lists
        stage_launches += 1
        merge_launches += 1

    return cur, stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    BLOCK_SIZE = 2048
    LOG_BLOCK = 11
    stage_warps = 8
    merge_warps = 16
    num_stages = 4

    use_merge_path = (
        (not _STATE.get("disable_merge", False))
        and (k >= 256)
        and (k <= 1024)
        and ((k & (k - 1)) == 0)
    )

    if use_merge_path:
        try:
            output, stage_launches, merge_launches, MERGE_SIZE, LOG_MERGE = _run_merge_path(
                x, N, k, BLOCK_SIZE, LOG_BLOCK, stage_warps, merge_warps, num_stages
            )
            _LAST_CFG.clear()
            _LAST_CFG.update({
                "BLOCK_SIZE": BLOCK_SIZE,
                "LOG_BLOCK": LOG_BLOCK,
                "MERGE_SIZE": MERGE_SIZE,
                "LOG_MERGE": LOG_MERGE,
                "stage_warps": stage_warps,
                "merge_warps": merge_warps,
                "num_stages": num_stages,
                "k": k,
                "merge_path": 1,
                "fallback": 0,
                "stage_launches": stage_launches,
                "merge_launches": merge_launches,
            })
            return output
        except Exception:
            _STATE["disable_merge"] = True

    output, stage_launches = _run_iterative_fullsort(
        x, N, k, BLOCK_SIZE, LOG_BLOCK, stage_warps, num_stages
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "LOG_BLOCK": LOG_BLOCK,
        "stage_warps": stage_warps,
        "merge_warps": 0,
        "num_stages": num_stages,
        "k": k,
        "merge_path": 0,
        "fallback": 1 if use_merge_path else 0,
        "stage_launches": stage_launches,
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
```
