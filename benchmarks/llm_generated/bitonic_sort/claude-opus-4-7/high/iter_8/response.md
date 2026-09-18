Looking at the trajectory:
- Best triton was iter 6 (CHUNK=4096, 132.5×) — iter 7 regressed slightly
- Best cutile was iter 4 (BLOCK=1024, 54.7×) — iter 7 used BLOCK=2048 with no real gain

For Triton: revert to iter 6's CHUNK=4096 strategy but bump BLOCK_STEP to 4096 with more warps for better memory throughput on global steps.

For cuTile: the simple step-per-launch approach is severely launch-overhead bound. Port the Triton fused architecture (initial local sort + global steps + fused local merge) using reshape/extract/cat tricks to do compare-exchanges in registers.

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
    BLOCK_STEP = 4096
    num_warps_local = 16
    num_warps_step = 8

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
                num_warps=num_warps_step, num_stages=3,
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
def _local_sort(data, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(data, index=(pid,), shape=(CHUNK,))
    base = pid * CHUNK

    for kp in range(1, LOG2_CHUNK + 1):
        K_VAL = 1 << kp
        for jp in range(kp - 1, -1, -1):
            J_VAL = 1 << jp
            TWO_J = J_VAL * 2
            NUM_G = CHUNK // TWO_J

            t3 = ct.reshape(tile, (NUM_G, 2, J_VAL))
            low = ct.extract(t3, (0, 0, 0), shape=(NUM_G, 1, J_VAL))
            high = ct.extract(t3, (0, 1, 0), shape=(NUM_G, 1, J_VAL))
            mn = ct.minimum(low, high)
            mx = ct.maximum(low, high)

            g_arange = ct.arange(NUM_G, dtype=np.int32)
            offs_g = base + g_arange * TWO_J
            asc = (offs_g & K_VAL) == 0
            asc_3d = ct.reshape(asc, (NUM_G, 1, 1))
            asc_b = ct.broadcast_to(asc_3d, (NUM_G, 1, J_VAL))

            new_low = ct.where(asc_b, mn, mx)
            new_high = ct.where(asc_b, mx, mn)
            combined = ct.cat((new_low, new_high), axis=1)
            tile = ct.reshape(combined, (CHUNK,))

    ct.store(data, index=(pid,), tile=tile)


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


@ct.kernel
def _local_merge(data, k_val, CHUNK: ConstInt, LOG2_CHUNK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(data, index=(pid,), shape=(CHUNK,))
    base = pid * CHUNK

    for jp in range(LOG2_CHUNK - 1, -1, -1):
        J_VAL = 1 << jp
        TWO_J = J_VAL * 2
        NUM_G = CHUNK // TWO_J

        t3 = ct.reshape(tile, (NUM_G, 2, J_VAL))
        low = ct.extract(t3, (0, 0, 0), shape=(NUM_G, 1, J_VAL))
        high = ct.extract(t3, (0, 1, 0), shape=(NUM_G, 1, J_VAL))
        mn = ct.minimum(low, high)
        mx = ct.maximum(low, high)

        g_arange = ct.arange(NUM_G, dtype=np.int32)
        offs_g = base + g_arange * TWO_J
        asc = (offs_g & k_val) == 0
        asc_3d = ct.reshape(asc, (NUM_G, 1, 1))
        asc_b = ct.broadcast_to(asc_3d, (NUM_G, 1, J_VAL))

        new_low = ct.where(asc_b, mn, mx)
        new_high = ct.where(asc_b, mx, mn)
        combined = ct.cat((new_low, new_high), axis=1)
        tile = ct.reshape(combined, (CHUNK,))

    ct.store(data, index=(pid,), tile=tile)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    CHUNK = 1024
    LOG2_CHUNK = 10
    BLOCK_STEP = 1024

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * CHUNK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_chunks = M // CHUNK
    n_pairs = M // 2
    grid_local = (n_chunks, 1, 1)
    grid_step = (max(n_pairs // BLOCK_STEP, 1), 1, 1)

    ct.launch(stream, grid_local, _local_sort, (work, CHUNK, LOG2_CHUNK))

    k = 2 * CHUNK
    while k <= M:
        j = k // 2
        while j >= CHUNK:
            ct.launch(stream, grid_step, _bitonic_step, (work, k, j, BLOCK_STEP))
            j //= 2
        ct.launch(stream, grid_local, _local_merge, (work, k, CHUNK, LOG2_CHUNK))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "CHUNK": CHUNK,
        "BLOCK_STEP": BLOCK_STEP,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
