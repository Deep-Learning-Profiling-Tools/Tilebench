Looking at the trajectory:
- Triton at 0.2% roofline is dominated by 5 sort stages with CHUNK=4096; need larger CHUNK to reduce stage count.
- cuTile failed because `with_hints` doesn't exist on the kernel object in this version — use the `@ct.kernel(occupancy=...)` decorator form instead.

I'll bump CHUNK to 8192 in Triton (4 stages for the largest case) and fix the cuTile launch by setting occupancy via the decorator.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _topk_stage_kernel(in_ptr, out_ptr, N_IN,
                       K: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < N_IN
    x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'))
    sorted_x = tl.sort(x, descending=True)
    idx = tl.arange(0, CHUNK)
    store_offs = pid * K + idx
    tl.store(out_ptr + store_offs, sorted_x, mask=idx < K)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    CHUNK = 8192
    num_warps = 8
    num_stages = 2

    stages = []
    cur_size = N
    while True:
        num_blocks = (cur_size + CHUNK - 1) // CHUNK
        if num_blocks <= 1:
            stages.append((cur_size, k, 1))
            break
        next_size = num_blocks * k
        stages.append((cur_size, next_size, num_blocks))
        cur_size = next_size

    max_inter = k
    for _, out_size, num_blocks in stages:
        if num_blocks > 1:
            max_inter = max(max_inter, out_size)

    if len(stages) > 1:
        buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)
    else:
        buf_a = buf_b = None

    src = input
    use_a = True
    for in_size, out_size, num_blocks in stages:
        if num_blocks == 1:
            dst = output
        else:
            dst = buf_a if use_a else buf_b
        _topk_stage_kernel[(num_blocks,)](
            src, dst, in_size,
            K=k, CHUNK=CHUNK,
            num_warps=num_warps, num_stages=num_stages,
        )
        src = dst
        use_a = not use_a

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK, "num_warps": num_warps, "num_stages": num_stages})
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

_CHUNK = 4096
_LOG_CHUNK = 12  # log2(_CHUNK)


@ct.kernel(occupancy=2)
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Bitonic sort descending: even-indexed subblocks sort descending,
    # odd-indexed ones ascending — after the final stage with nrows=1,
    # the entire tile ends up sorted descending.
    for stage_log in range(1, _LOG_CHUNK + 1):
        stage_k = 1 << stage_log
        for sub_log in range(stage_log - 1, -1, -1):
            sub_j = 1 << sub_log
            nrows = _CHUNK // stage_k
            npairs = stage_k // (2 * sub_j)

            a = ct.reshape(x, (nrows, npairs, 2, sub_j))
            mn = ct.min(a, axis=2, keepdims=True)
            mx = ct.max(a, axis=2, keepdims=True)

            row_idx = ct.arange(nrows, dtype=np.int32)
            is_odd = (row_idx & 1) != 0
            is_odd_4d = ct.reshape(is_odd, (nrows, 1, 1, 1))
            is_odd_b = ct.broadcast_to(is_odd_4d, (nrows, npairs, 1, sub_j))

            new_lo = ct.where(is_odd_b, mn, mx)
            new_hi = ct.where(is_odd_b, mx, mn)
            new_a = ct.cat((new_lo, new_hi), axis=2)
            x = ct.reshape(new_a, (_CHUNK,))

    top_k = ct.extract(x, (0,), (K,))
    ct.store(out_arr, index=(pid,), tile=top_k)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    CHUNK = _CHUNK

    stages = []
    cur_size = N
    while True:
        num_blocks = (cur_size + CHUNK - 1) // CHUNK
        if num_blocks <= 1:
            stages.append((cur_size, k, 1))
            break
        next_size = num_blocks * k
        stages.append((cur_size, next_size, num_blocks))
        cur_size = next_size

    max_inter = k
    for _, out_size, num_blocks in stages:
        if num_blocks > 1:
            max_inter = max(max_inter, out_size)

    if len(stages) > 1:
        buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)
    else:
        buf_a = buf_b = None

    stream = torch.cuda.current_stream()

    src = input
    use_a = True
    for in_size, out_size, num_blocks in stages:
        if num_blocks == 1:
            dst = output
        else:
            base = buf_a if use_a else buf_b
            dst = base[:out_size]

        if src.numel() != in_size:
            src_view = src[:in_size]
        else:
            src_view = src

        ct.launch(stream, (num_blocks, 1, 1), _topk_stage_kernel, (src_view, dst, k))
        src = dst
        use_a = not use_a

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
