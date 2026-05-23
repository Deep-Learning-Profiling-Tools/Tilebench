Looking at the trajectory:
- iter 5 (best verify-clean): TILE=16384 single-pass, 53.7%
- iter 7 (broke verify): multi-tile keep-in-registers approach was actually a good idea but had `F.normalize` in a comment, triggering the forbidden-pattern guard

The multi-tile approach avoids the padding waste of TILE=16384 for K=10240 (where ~37% of loads are wasted padding). Re-emitting that approach **with no forbidden-pattern strings anywhere in the source**, using TILE=2048 × NUM_TILES=5 for the largest case (perfect coverage of K=10240), keeping all tiles live in registers between sum-of-squares and scaled store.

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

    # Single sweep over the row: load each sub-tile in fp32, keep it live in
    # registers (the python loop is fully unrolled at compile time, so the
    # tiles end up as distinct SSA values held in regs across the sum step),
    # accumulate the partial sum-of-squares.
    xs = []
    acc = 0.0
    for j in range(NUM_TILES):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        xs.append(xj)
        acc = acc + ct.sum(xj * xj)

    # Torch reference semantics: y = x / max(||x||_2, eps)
    inv = 1.0 / ct.maximum(ct.sqrt(acc), eps)

    # Pass 2 — purely register → DRAM, no extra global reads.
    for j in range(NUM_TILES):
        yj = ct.astype(xs[j] * inv, x.dtype)
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

    # Tile selection:
    #   K <= 2048  -> single tile sized to next pow2 (covers 512,1024,2048).
    #   K  > 2048  -> TILE=2048, NUM_TILES = cdiv(K, 2048). For the timing
    #                 case K=10240 this gives NUM_TILES=5 with zero padding
    #                 waste (vs ~37% waste at TILE=16384 single-pass).
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
        "single_pass_register_tiles": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: revert to iter-7's multi-tile single-pass strategy (load all sub-tiles, accumulate, then scale-store from registers) but with the offending pattern-trigger comment removed. For K=10240, TILE=2048×5 eliminates the ~37% padding waste that TILE=16384 single-pass had, while still loading each element exactly once.
