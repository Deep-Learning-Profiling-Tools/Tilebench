import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _l2_norm_kernel(x, out, eps: float,
                    K: ConstInt, TILE: ConstInt, NUM_TILES: ConstInt):
    row = ct.bid(0)

    # Pass 1: load all tiles, accumulate sum-of-squares in fp32.
    # Tiles stay in registers (NUM_TILES is compile-time → unrolled).
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

    # F.normalize uses: rsqrt(max(sum_sq, eps^2)) which equals 1/max(norm, eps)
    inv = 1.0 / ct.maximum(ct.sqrt(acc), eps)

    # Pass 2: scale and store.
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

    # Pick TILE so that NUM_TILES * TILE covers K with minimal waste.
    # For K=10240 (largest case), TILE=2048 → NUM_TILES=5, zero waste.
    # For smaller K (512..2048), TILE=2048 with NUM_TILES=1 (one tile).
    # For K in (2048, 4096], NUM_TILES=2; etc.
    # TILE must be power of 2 and ≥ 256.
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
        "single_pass_multi_tile": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
