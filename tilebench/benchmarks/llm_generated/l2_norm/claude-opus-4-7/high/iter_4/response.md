Looking at the trajectory:
- Triton iter 1 (BLOCK_K=2048, num_warps=8, num_stages=2) was best at 76% — added LOOP_STAGES in iter 3 regressed
- cuTile improving with larger tiles (TILE=4096 occupancy=2 at 50%)

For Triton, the main remaining inefficiency at 76% is likely **wave imbalance**: M=2048 CTAs running ~4 per SM on 148 SMs gives ~3.5 waves with the last wave half-empty. A **persistent kernel** with grid sized to the SM count fixes this.

For cuTile, push the tile size further to amortize loop overhead.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}

_NUM_SM = None
def _num_sm():
    global _NUM_SM
    if _NUM_SM is None:
        _NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SM


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, M, K, eps, stride_row,
                    BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    for row in tl.range(pid, M, num_progs):
        x_row = x_ptr + row * stride_row
        o_row = out_ptr + row * stride_row

        # Pass 1: sum of squares (fp32). evict_last keeps data in L2 for pass 2.
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0,
                        eviction_policy="evict_last").to(tl.float32)
            acc += x * x
        total = tl.sum(acc, axis=0)
        inv = 1.0 / tl.maximum(tl.sqrt(total), eps)

        # Pass 2: normalize. evict_first since we're done with this row.
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)
            y = x * inv
            tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    BLOCK_K = 2048
    num_warps = 8
    num_stages = 2

    # Persistent grid: NUM_SM * occupancy programs, each handling multiple rows.
    # This eliminates wave imbalance (2048 rows on 148 SMs would have a half-empty
    # final wave).
    grid_size = min(M, _num_sm() * 4)
    grid = (grid_size,)

    _l2_norm_kernel[grid](
        x_flat, out_flat, M, K, float(eps), x_flat.stride(0),
        BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "grid_size": grid_size,
        "persistent": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _l2_norm_kernel(x, out, eps: float, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(K, TILE)

    # Pass 1: sum of squares in fp32 (keep data in L2 for pass 2).
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=10),
            np.float32,
        )
        acc = acc + xj * xj

    total = ct.sum(acc)
    inv = 1.0 / ct.maximum(ct.sqrt(total), eps)

    # Pass 2: normalize and write back.
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        yj = xj * inv
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype),
                 latency=1)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    TILE = 8192
    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel,
              (x_flat, out_flat, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 2})
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Approach:** Triton — keep iter 1's winning inner config (BLOCK_K=2048, num_warps=8, num_stages=2) but switch to a **persistent kernel** sized to `NUM_SM × 4` to eliminate the half-empty final wave that caps non-persistent launches at ~76%. cuTile — bump TILE to 8192 (only 2 iters per pass for K=10240) to reduce loop overhead while keeping occupancy=2.
