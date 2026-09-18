Looking at the trajectory, both backends are stuck at ~0.2% roofline with ~60-70× speedup vs torch. The bottleneck is that each compare-exchange step does very little work (~3 ops) per element moved (~16 bytes for fp32 = pair load+store). With ~log²M passes, total memory traffic is `O(log²M × M)` while the roofline assumes `2N`.

**Key optimization: fuse multiple bitonic steps per kernel launch using in-tile permutations.**

- When step distance `j < CHUNK_SIZE`, all pairs `(i, i^j)` lie in the same CHUNK-sized chunk → one CTA can perform many `j`-steps without touching DRAM between them.
- Triton: use `tl.gather` for tile-internal permutation.
- cuTile: use `reshape → extract → cat` since `ct.gather` only operates on Arrays.
- This cuts memory traffic by ~3× and launch count from ~300 to ~105.

Three kernels each: (1) initial local sort (all k≤CHUNK fused), (2) cross-CTA step (for j≥CHUNK), (3) fused local merge (all j<CHUNK for a given outer k).

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

    CHUNK = 2048
    LOG2_CHUNK = 11
    BLOCK_STEP = 1024
    num_warps_local = 8
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
def _initial_local_sort_ct(data, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    base = pid * CHUNK
    x = ct.load(data, index=(pid,), shape=(CHUNK,))

    for kp in range(1, LOG2_CHUNK + 1):
        k_v = 1 << kp
        for jp in range(kp - 1, -1, -1):
            j = 1 << jp
            G = CHUNK // (2 * j)
            x_r = x.reshape((G, 2, j))
            lo = ct.extract(x_r, (0, 0, 0), (G, 1, j)).reshape((G, j))
            hi = ct.extract(x_r, (0, 1, 0), (G, 1, j)).reshape((G, j))
            mn = ct.minimum(lo, hi)
            mx = ct.maximum(lo, hi)
            group_offs = ct.arange(G, dtype=np.int32) * (2 * j) + base
            asc_g = (group_offs & k_v) == 0
            new_lo = ct.where(asc_g[:, None], mn, mx)
            new_hi = ct.where(asc_g[:, None], mx, mn)
            combined = ct.cat(
                (new_lo.reshape((G, 1, j)), new_hi.reshape((G, 1, j))),
                axis=1,
            )
            x = combined.reshape((CHUNK,))

    ct.store(data, index=(pid,), tile=x)


@ct.kernel
def _step_kernel_ct(data, k_val, j_val, BLOCK: ConstInt):
    pid = ct.bid(0)
    tid = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = (tid // j_val) * (2 * j_val) + (tid % j_val)
    ixj = i + j_val
    ai = ct.gather(data, i, check_bounds=False)
    aj = ct.gather(data, ixj, check_bounds=False)
    asc = (i & k_val) == 0
    mn = ct.minimum(ai, aj)
    mx = ct.maximum(ai, aj)
    ct.scatter(data, i, ct.where(asc, mn, mx), check_bounds=False)
    ct.scatter(data, ixj, ct.where(asc, mx, mn), check_bounds=False)


@ct.kernel
def _fused_local_merge_ct(data, k_val, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    base = pid * CHUNK
    x = ct.load(data, index=(pid,), shape=(CHUNK,))
    asc = (base & k_val) == 0  # scalar bool: uniform within chunk (since k_val > CHUNK)

    for jp in range(LOG2_CHUNK - 1, -1, -1):
        j = 1 << jp
        G = CHUNK // (2 * j)
        x_r = x.reshape((G, 2, j))
        lo = ct.extract(x_r, (0, 0, 0), (G, 1, j)).reshape((G, j))
        hi = ct.extract(x_r, (0, 1, 0), (G, 1, j)).reshape((G, j))
        mn = ct.minimum(lo, hi)
        mx = ct.maximum(lo, hi)
        new_lo = ct.where(asc, mn, mx)
        new_hi = ct.where(asc, mx, mn)
        combined = ct.cat(
            (new_lo.reshape((G, 1, j)), new_hi.reshape((G, 1, j))),
            axis=1,
        )
        x = combined.reshape((CHUNK,))

    ct.store(data, index=(pid,), tile=x)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    CHUNK = 1024
    LOG2_CHUNK = 10
    BLOCK_STEP = 1024

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, CHUNK, 2 * BLOCK_STEP)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_chunks = M // CHUNK
    n_pairs = M // 2
    n_step_blocks = max(n_pairs // BLOCK_STEP, 1)

    ct.launch(stream, (n_chunks, 1, 1),
              _initial_local_sort_ct, (work, CHUNK, LOG2_CHUNK))

    k = 2 * CHUNK
    while k <= M:
        j = k // 2
        while j >= CHUNK:
            ct.launch(stream, (n_step_blocks, 1, 1),
                      _step_kernel_ct, (work, k, j, BLOCK_STEP))
            j //= 2
        ct.launch(stream, (n_chunks, 1, 1),
                  _fused_local_merge_ct, (work, k, CHUNK, LOG2_CHUNK))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK, "BLOCK_STEP": BLOCK_STEP})
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
