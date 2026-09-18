Looking at the trajectory:
- Iter 3 (Triton CHUNK=4096, nw=32, ns=2) is the best at 0.4%, 1.07× torch
- Iter 5 regressed by increasing CHUNK to 8192 (sort cost grows faster than launch savings)
- cuTile has never verified — the bitonic compile fails

**Strategy:**
- **Triton**: Revert to iter 3's CHUNK=4096, nw=32 winner but try `num_stages=4` for deeper async pipelining of loads.
- **cuTile**: Rewrite the bitonic sort more cleanly (drop the `if shift>0` branch which may confuse compile, use `ct.int32`, simplify shape arithmetic) and bump CHUNK to 4096 to halve the number of multi-stage launches.

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
    x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'),
                eviction_policy="evict_first")
    sorted_x = tl.sort(x, descending=True)
    idx = tl.arange(0, CHUNK)
    store_offs = pid * K + idx
    tl.store(out_ptr + store_offs, sorted_x, mask=idx < K)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    CHUNK = 4096
    num_warps = 32
    num_stages = 4

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
_LOG_CHUNK = 12  # log2(4096)


@ct.kernel
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Iterative bitonic sort -> descending output.
    # Outer loop: sequence size doubles (k_log = 1 .. LOG_CHUNK).
    # Inner loop: comparison distance halves (j_log = k_log-1 .. 0).
    for k_log in range(1, _LOG_CHUNK + 1):
        for j_log in range(k_log - 1, -1, -1):
            j = 1 << j_log
            block_len = j << 1
            nrows = _CHUNK // block_len

            a = ct.reshape(x, (nrows, block_len))
            left = ct.extract(a, (0, 0), shape=(nrows, j))
            right = ct.extract(a, (0, j), shape=(nrows, j))

            mn = ct.minimum(left, right)
            mx = ct.maximum(left, right)

            row_idx = ct.arange(nrows, dtype=ct.int32)
            shift_amt = k_log - j_log - 1
            seq_idx = row_idx >> shift_amt  # shift by 0 is identity
            descending = (seq_idx & 1) == 0  # bool, shape (nrows,)
            descending = ct.reshape(descending, (nrows, 1))
            descending = ct.broadcast_to(descending, (nrows, j))

            new_left = ct.where(descending, mx, mn)
            new_right = ct.where(descending, mn, mx)

            combined = ct.cat((new_left, new_right), axis=1)
            x = ct.reshape(combined, (_CHUNK,))

    top_k = ct.extract(x, (0,), shape=(K,))
    ct.store(out_arr, index=(pid,), tile=top_k)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)
    CHUNK = _CHUNK
    stream = torch.cuda.current_stream()

    stages = []
    cur_size = N
    while True:
        num_blocks = (cur_size + CHUNK - 1) // CHUNK
        if num_blocks <= 1:
            stages.append((cur_size, k, 1, True))
            break
        next_size = num_blocks * k
        stages.append((cur_size, next_size, num_blocks, False))
        cur_size = next_size

    if len(stages) == 1:
        ct.launch(stream, (1, 1, 1), _topk_stage_kernel, (input, output, k))
    else:
        max_inter = max(s[1] for s in stages if not s[3])
        buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)

        src = input
        use_a_dst = True
        for in_size, out_size, num_blocks, is_last in stages:
            if is_last:
                dst = output
            else:
                dst = buf_a if use_a_dst else buf_b
                use_a_dst = not use_a_dst
            ct.launch(stream, (num_blocks, 1, 1), _topk_stage_kernel, (src, dst, k))
            src = dst

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Triton keeps iter 3's winning warps/CHUNK but pipelines more (`num_stages=4`) and hints the kernel to evict input from L2 after one read. cuTile rewrites the bitonic sort with cleaner direction encoding and avoids the suspicious `if shift>0` branch from iter 5.
