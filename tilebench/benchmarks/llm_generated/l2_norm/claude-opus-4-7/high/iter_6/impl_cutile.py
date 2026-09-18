import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _l2_norm_kernel(x, out, eps: float,
                    K: ConstInt, TILE: ConstInt, NUM_TILES: ConstInt):
    row = ct.bid(0)

    # Single-pass, fully unrolled: load every sub-tile of the row, compute
    # sum-of-squares, then store normalized output. Tiles stay in registers
    # between the two phases — x is read once, y is written once.
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

    inv = 1.0 / ct.maximum(ct.sqrt(acc), eps)

    for j in range(NUM_TILES):
        yj = ct.astype(xs[j] * inv, x.dtype)
        ct.store(out, index=(row, j), tile=yj)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    # TILE=2048 evenly divides K=10240 (the largest case) → zero padding waste.
    # NUM_TILES is ConstInt (compile-time), so the for-loops above unroll and
    # every loaded sub-tile lives in registers between Pass-1 and Pass-2.
    TILE = 2048
    NUM_TILES = (K + TILE - 1) // TILE
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel,
              (x_flat, out_flat, float(eps), K, TILE, NUM_TILES))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "NUM_TILES": NUM_TILES,
        "occupancy": occupancy,
        "single_pass_unrolled": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
