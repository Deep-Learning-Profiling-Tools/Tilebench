Looking at the trajectory, my Triton implementation is at only 0.4% of roofline. The fundamental issue is that the multi-stage sort approach requires reading the input data ~3-4× total (multiple stages each reading the previous stage's output). The roofline calculation assumes only one pass.

**Strategy switch — persistent top-k kernel:**
- Each block maintains a "running top-K" tile of size 2K (top K in first K positions, -inf elsewhere)
- Each iteration loads K=LOAD new elements into the second half of the working tile, uses element-wise `tl.maximum` (which acts as a concat trick because the empty slots are -inf), sorts descending, keeps first K
- This achieves O(N) memory reads in stage 1, much closer to roofline
- Multi-stage: stage 1 with many persistent blocks, then progressively fewer blocks until 1

For cuTile: keep the multi-stage bitonic sort but fix likely issues — use explicit `broadcast_to`, remove tensor slicing (which caused different-shape recompiles), use clean operator forms.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}

_TILE = 2048
_LOAD = 1024


@triton.jit
def _topk_persistent_kernel(in_ptr, out_ptr, N,
                            K: tl.constexpr,
                            TILE: tl.constexpr,
                            LOAD: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    NEG_INF: tl.constexpr = float('-inf')

    running = tl.full((TILE,), NEG_INF, dtype=tl.float32)
    idx = tl.arange(0, TILE)

    elems_per_block = tl.cdiv(N, num_progs)
    start = pid * elems_per_block
    end = tl.minimum(start + elems_per_block, N)

    # Process LOAD elements per iteration; the working tile has size TILE = K + LOAD' >= 2K.
    # Layout: positions [0,LOAD) hold previous-iter top-K (rest -inf), positions [LOAD,TILE)
    # receive newly-loaded elements. After `tl.maximum`, the tile equals concat(running_first_K, new_data).
    for offset in range(start, end, LOAD):
        new_offs = offset + (idx - LOAD)
        valid = (idx >= LOAD) & (new_offs < end)
        new_tile = tl.load(in_ptr + new_offs, mask=valid, other=NEG_INF)
        combined = tl.maximum(running, new_tile)
        sorted_c = tl.sort(combined, descending=True)
        running = tl.where(idx < K, sorted_c, NEG_INF)

    # Store first K of running to out_ptr[pid*K : pid*K + K]. Mask suppresses the extra TILE-K positions.
    out_offs = pid * K + idx
    out_mask = idx < K
    tl.store(out_ptr + out_offs, running, mask=out_mask)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    assert 1 <= k <= N
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    TILE = _TILE
    LOAD = _LOAD
    num_warps = 8
    num_stages = 2
    target_per_block = 8192  # tune: elements per block per stage

    cur = input
    cur_size = N

    intermediates = []
    while True:
        num_persist = max(1, (cur_size + target_per_block - 1) // target_per_block)
        num_persist = min(num_persist, 256)

        if num_persist == 1:
            _topk_persistent_kernel[(1,)](
                cur, output, cur_size,
                K=k, TILE=TILE, LOAD=LOAD,
                num_warps=num_warps, num_stages=num_stages,
            )
            break

        intermediate = torch.empty(num_persist * k, dtype=torch.float32, device=input.device)
        intermediates.append(intermediate)
        _topk_persistent_kernel[(num_persist,)](
            cur, intermediate, cur_size,
            K=k, TILE=TILE, LOAD=LOAD,
            num_warps=num_warps, num_stages=num_stages,
        )
        cur = intermediate
        cur_size = num_persist * k

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE, "LOAD": LOAD,
        "target_per_block": target_per_block,
        "num_warps": num_warps, "num_stages": num_stages,
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

_CHUNK = 4096
_LOG_CHUNK = 12  # log2(4096)


@ct.kernel(occupancy=2)
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Bitonic sort, descending. View x as (nrows, 2, sub_j) at each step.
    for stage_log in range(1, _LOG_CHUNK + 1):
        for sub_log in range(stage_log - 1, -1, -1):
            sub_j = 1 << sub_log
            block_len = 2 * sub_j
            nrows = _CHUNK // block_len
            shift = stage_log - sub_log - 1

            a = ct.reshape(x, (nrows, block_len))
            left = ct.extract(a, (0, 0), shape=(nrows, sub_j))
            right = ct.extract(a, (0, sub_j), shape=(nrows, sub_j))

            mn = ct.minimum(left, right)
            mx = ct.maximum(left, right)

            row_idx = ct.arange(nrows, dtype=np.int32)
            if shift > 0:
                shifted = ct.bitwise_rshift(row_idx, shift)
            else:
                shifted = row_idx
            parity = ct.bitwise_and(shifted, 1)
            is_desc_1d = parity == 0
            is_desc_2d = ct.reshape(is_desc_1d, (nrows, 1))
            is_desc_b = ct.broadcast_to(is_desc_2d, (nrows, sub_j))

            new_left = ct.where(is_desc_b, mx, mn)
            new_right = ct.where(is_desc_b, mn, mx)

            new_a = ct.cat((new_left, new_right), axis=1)
            x = ct.reshape(new_a, (_CHUNK,))

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
        # Single launch directly to output
        ct.launch(stream, (1, 1, 1), _topk_stage_kernel, (input, output, k))
    else:
        # Allocate two ping-pong buffers, same size, never sliced -> minimises shape variations.
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
    _LAST_CFG.update({"CHUNK": CHUNK, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: For Triton I switched from full-sort multi-stage to a **persistent top-K kernel** that maintains a running top-K tile and merges new chunks via a `tl.maximum`+`tl.sort` concat trick, reducing the total memory traffic close to the optimal N+k bytes. For cuTile I kept the multi-stage bitonic sort but cleaned up broadcasting (explicit `broadcast_to`), used `ct.bitwise_and` explicitly, and removed tensor slicing across launches to minimize shape-driven recompiles that likely caused the iter-3 compile failure on the largest case.
