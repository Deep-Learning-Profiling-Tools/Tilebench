Looking at this, my iter 7 best was 1.07× with CHUNK=4096 direct sort. The bottleneck for the largest case (N=1M, k=1024) is that direct sort needs many stages (5) due to small reduction ratio (CHUNK/k=4).

**New approach: streaming top-K via incremental merge-sort.** Each block sequentially loads K elements, concatenates with running top-K [K], sorts [2K], extracts first K. This reduces N=1M, k=1024 to 2 launches with reads ≈ N+32K ≈ optimal bandwidth.

For small k (≤256), the direct approach remains better (smaller k means streaming has more sort overhead per element). For cuTile, the previous bitonic sort approach kept failing to compile — falling back to `NotImplementedError` so the harness skips it cleanly.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _topk_streaming_kernel(in_ptr, out_ptr, N_IN,
                           ELEMS_PER_BLOCK: tl.constexpr,
                           K: tl.constexpr):
    """Each block scans ELEMS_PER_BLOCK elements maintaining running top-K via merge-sort."""
    pid = tl.program_id(0)
    start = pid * ELEMS_PER_BLOCK

    running = tl.full([K], -float('inf'), dtype=tl.float32)

    for i in range(0, ELEMS_PER_BLOCK // K):
        offs = start + i * K + tl.arange(0, K)
        mask = offs < N_IN
        x_small = tl.load(in_ptr + offs, mask=mask, other=-float('inf'),
                          eviction_policy="evict_first")
        combined = tl.cat(running, x_small)                          # [2K]
        sorted_combined = tl.sort(combined, descending=True)
        sorted_2d = tl.reshape(sorted_combined, (2, K))              # row 0 = first K
        sorted_2d_t = tl.permute(sorted_2d, (1, 0))                  # [K, 2]
        first_half, _ = tl.split(sorted_2d_t)                        # [K]
        running = first_half

    out_offs = pid * K + tl.arange(0, K)
    tl.store(out_ptr + out_offs, running)


@triton.jit
def _topk_direct_kernel(in_ptr, out_ptr, N_IN,
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

    if k >= 1024:
        # Streaming approach: 2-stage reduction with near-optimal bandwidth.
        MAX_ELEMS = 32768
        num_warps = 32
        num_stages = 2

        src = input
        cur_size = N

        while True:
            ELEMS = min(MAX_ELEMS, cur_size)
            # ELEMS divisible by k since both are pow2 with ELEMS >= k.
            num_blocks = (cur_size + ELEMS - 1) // ELEMS

            if num_blocks <= 1:
                _topk_streaming_kernel[(1,)](
                    src, output, cur_size,
                    ELEMS_PER_BLOCK=ELEMS, K=k,
                    num_warps=num_warps, num_stages=num_stages,
                )
                break

            next_size = num_blocks * k
            dst = torch.empty(next_size, dtype=torch.float32, device=input.device)
            _topk_streaming_kernel[(num_blocks,)](
                src, dst, cur_size,
                ELEMS_PER_BLOCK=ELEMS, K=k,
                num_warps=num_warps, num_stages=num_stages,
            )
            src = dst
            cur_size = next_size

        _LAST_CFG.clear()
        _LAST_CFG.update({"MAX_ELEMS": MAX_ELEMS, "K": k,
                          "num_warps": num_warps, "num_stages": num_stages,
                          "strategy": "streaming"})
    else:
        # Direct sort approach for small k (verified-best from iter 3).
        CHUNK = 4096
        num_warps = 32
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
            _topk_direct_kernel[(num_blocks,)](
                src, dst, in_size,
                K=k, CHUNK=CHUNK,
                num_warps=num_warps, num_stages=num_stages,
            )
            src = dst
            use_a = not use_a

        _LAST_CFG.clear()
        _LAST_CFG.update({"CHUNK": CHUNK, "K": k,
                          "num_warps": num_warps, "num_stages": num_stages,
                          "strategy": "direct"})

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch

_LAST_CFG: dict | None = None


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    # cuTile lacks a built-in sort primitive; hand-rolled bitonic sort
    # via reshape/extract/cat/where consistently fails to compile on the
    # toolchain. Skip cuTile cleanly so it doesn't block triton.
    raise NotImplementedError(
        "cuTile top_k_selection: bitonic sort variants fail to compile; "
        "skipping until cuTile exposes a sort primitive."
    )


def get_last_config() -> dict | None:
    return None
```
