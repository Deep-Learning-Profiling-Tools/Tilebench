Looking at this:
- Iter 5 (best): single-pass TILE=16384 → 53.7%. For K=10240 this wastes ~37% bandwidth on padding.
- Iter 8 broke compile trying to cache tiles in a Python list across passes.

Safer optimization: classic two-pass kernel with TILE=2048 (NUM_TILES=5 for K=10240, **zero padding waste**). Re-reads x once (2 reads + 1 write = 3·K·dtype bytes), vs single-pass with TILE=16384 reading 16384/10240 ≈ 1.6× padded data per pass (≈3.2·K). So two-pass without padding waste actually moves *less* DRAM traffic for K=10240.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _l2_norm_kernel(x, out, eps: float,
                    K: ConstInt, TILE: ConstInt, NUM_TILES: ConstInt):
    row = ct.bid(0)

    # Pass 1: accumulate sum of squares in fp32.
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(NUM_TILES):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        acc = acc + xj * xj

    total = ct.sum(acc)
    inv = 1.0 / ct.maximum(ct.sqrt(total), eps)

    # Pass 2: re-load and scale.
    for j in range(NUM_TILES):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = ct.astype(xj * inv, x.dtype)
        ct.store(out, index=(row, j), tile=yj)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    # Tile sizing:
    #  - small rows (K<=2048): single-pass with TILE = next_pow2(K), no waste.
    #  - large rows (K>2048): multi-tile pass with TILE=2048. For K=10240,
    #    NUM_TILES=5 with zero padding waste (vs ~37% waste at TILE=16384).
    if K <= 2048:
        TILE = max(_next_pow2(K), 256)
    else:
        TILE = 2048
    NUM_TILES = (K + TILE - 1) // TILE

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel,
              (x_flat, out_flat, float(eps), K, TILE, NUM_TILES))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "NUM_TILES": NUM_TILES,
        "occupancy": 2,
        "strategy": "two_pass_no_padding_waste",
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Two-pass with TILE=2048 eliminates padding waste at the largest case (K=10240), reducing total DRAM bytes vs the single-pass TILE=16384 approach from iter 5 while keeping the kernel structurally simple (no register-resident tile lists that broke iter 8's compile).
