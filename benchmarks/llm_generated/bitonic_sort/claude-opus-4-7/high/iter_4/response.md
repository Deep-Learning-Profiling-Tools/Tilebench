Looking at the trajectory, my Triton kernel kept failing compilation (likely `tl.gather` issues with large constexpr blocks). The cutile path works at 0.2% which is actually near-optimal given the roofline is measured against `n * 2 * dtype_size` but bitonic sort fundamentally needs O(log²N) passes over the data.

Let me simplify Triton to mirror the working cutile approach (simple step-by-step) to first achieve verification, then add a local in-block sort optimization that avoids `tl.gather` by using explicit unrolled bitonic stages.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_step(data_ptr, k, j, n_pairs,
                  BLOCK: tl.constexpr):
    """One bitonic compare-exchange step at distance j within length-k subsequences."""
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_pairs
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = tl.load(data_ptr + i, mask=mask)
    aj = tl.load(data_ptr + ixj, mask=mask)
    asc = (i & k) == 0
    mn = tl.minimum(ai, aj)
    mx = tl.maximum(ai, aj)
    tl.store(data_ptr + i, tl.where(asc, mn, mx), mask=mask)
    tl.store(data_ptr + ixj, tl.where(asc, mx, mn), mask=mask)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK = 1024
    num_warps = 4
    num_stages = 2

    M = 1 << ((N - 1).bit_length())

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_pairs = M // 2
    grid = (triton.cdiv(n_pairs, BLOCK),)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            _bitonic_step[grid](
                work, k, j, n_pairs,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            j //= 2
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _bitonic_step(data, k, j, BLOCK: ConstInt):
    pid = ct.bid(0)
    tid = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = ct.gather(data, i, check_bounds=False)
    aj = ct.gather(data, ixj, check_bounds=False)
    asc = (i & k) == 0
    mn = ct.minimum(ai, aj)
    mx = ct.maximum(ai, aj)
    ct.scatter(data, i, ct.where(asc, mn, mx), check_bounds=False)
    ct.scatter(data, ixj, ct.where(asc, mx, mn), check_bounds=False)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK = 1024

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * BLOCK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_pairs = M // 2
    grid = (max(n_pairs // BLOCK, 1), 1, 1)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ct.launch(stream, grid, _bitonic_step, (work, k, j, BLOCK))
            j //= 2
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK": BLOCK})
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted Triton to a simple step-by-step kernel (matching the working cutile approach) to achieve verification first. Both now use `BLOCK=1024` with simple masked loads/stores per compare-exchange step.
