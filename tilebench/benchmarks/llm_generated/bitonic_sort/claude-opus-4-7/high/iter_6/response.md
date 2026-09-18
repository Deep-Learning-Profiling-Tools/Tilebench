Looking at the trajectory:
- **Triton**: iter 5 (CHUNK=2048 with fused local sort/merge) is the best at 0.4% / 129×. Note that bitonic sort fundamentally requires O(log²M) passes over the data, so peak roofline % ≈ 1/300 ≈ 0.33% — we're already exceeding the per-pass algorithmic ideal. I'll keep iter 5's structure but try CHUNK=4096 (LOG2_CHUNK=12) to reduce the number of global merge passes.
- **CuTile**: iter 5 broke compile due to the complex reshape/extract chain. Revert to iter 4's simple step kernel and try BLOCK=2048 with an occupancy hint as a smaller, safer change.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _initial_local_sort(data_ptr, CHUNK: tl.constexpr, LOG2_CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs_tile = tl.arange(0, CHUNK)
    offs = base + offs_tile
    x = tl.load(data_ptr + offs)

    for kp in tl.static_range(1, LOG2_CHUNK + 1):
        asc = (offs & (1 << kp)) == 0
        for jp in tl.static_range(kp - 1, -1, -1):
            j_val = 1 << jp
            partner_tile = offs_tile ^ j_val
            x_partner = tl.gather(x, partner_tile, axis=0)
            is_low = (offs_tile & j_val) == 0
            mn = tl.minimum(x, x_partner)
            mx = tl.maximum(x, x_partner)
            new_low = tl.where(asc, mn, mx)
            new_high = tl.where(asc, mx, mn)
            x = tl.where(is_low, new_low, new_high)

    tl.store(data_ptr + offs, x)


@triton.jit
def _step_kernel(data_ptr, k_val, j_val, n_pairs, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_pairs
    i = (tid // j_val) * (2 * j_val) + (tid % j_val)
    ixj = i + j_val
    ai = tl.load(data_ptr + i, mask=mask)
    aj = tl.load(data_ptr + ixj, mask=mask)
    asc = (i & k_val) == 0
    mn = tl.minimum(ai, aj)
    mx = tl.maximum(ai, aj)
    tl.store(data_ptr + i, tl.where(asc, mn, mx), mask=mask)
    tl.store(data_ptr + ixj, tl.where(asc, mx, mn), mask=mask)


@triton.jit
def _fused_local_merge(data_ptr, k_val, CHUNK: tl.constexpr, LOG2_CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs_tile = tl.arange(0, CHUNK)
    x = tl.load(data_ptr + base + offs_tile)
    asc = (base & k_val) == 0

    for jp in tl.static_range(LOG2_CHUNK - 1, -1, -1):
        j_val = 1 << jp
        partner_tile = offs_tile ^ j_val
        x_partner = tl.gather(x, partner_tile, axis=0)
        is_low = (offs_tile & j_val) == 0
        mn = tl.minimum(x, x_partner)
        mx = tl.maximum(x, x_partner)
        new_low = tl.where(asc, mn, mx)
        new_high = tl.where(asc, mx, mn)
        x = tl.where(is_low, new_low, new_high)

    tl.store(data_ptr + base + offs_tile, x)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    CHUNK = 4096
    LOG2_CHUNK = 12
    BLOCK_STEP = 2048
    num_warps_local = 16
    num_warps_step = 4

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, CHUNK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_chunks = M // CHUNK
    n_pairs = M // 2

    _initial_local_sort[(n_chunks,)](
        work, CHUNK=CHUNK, LOG2_CHUNK=LOG2_CHUNK,
        num_warps=num_warps_local, num_stages=1,
    )

    grid_step = (triton.cdiv(n_pairs, BLOCK_STEP),)

    k = 2 * CHUNK
    while k <= M:
        j = k // 2
        while j >= CHUNK:
            _step_kernel[grid_step](
                work, k, j, n_pairs,
                BLOCK=BLOCK_STEP,
                num_warps=num_warps_step, num_stages=2,
            )
            j //= 2
        _fused_local_merge[(n_chunks,)](
            work, k,
            CHUNK=CHUNK, LOG2_CHUNK=LOG2_CHUNK,
            num_warps=num_warps_local, num_stages=1,
        )
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "CHUNK": CHUNK,
        "BLOCK_STEP": BLOCK_STEP,
        "num_warps_local": num_warps_local,
        "num_warps_step": num_warps_step,
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

    BLOCK = 2048

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * BLOCK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_pairs = M // 2
    grid = (max(n_pairs // BLOCK, 1), 1, 1)

    kernel = _bitonic_step.with_hints(occupancy=4)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ct.launch(stream, grid, kernel, (work, k, j, BLOCK))
            j //= 2
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK": BLOCK, "occupancy": 4})
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: Triton keeps iter 5's structure (initial local sort + global step + fused local merge) but doubles CHUNK to 4096 to halve the number of global step launches at the cost of more registers. CuTile reverts to iter 4's verify-clean simple step kernel with a larger BLOCK (2048) and an explicit occupancy hint of 4 as a small, safe optimization.
