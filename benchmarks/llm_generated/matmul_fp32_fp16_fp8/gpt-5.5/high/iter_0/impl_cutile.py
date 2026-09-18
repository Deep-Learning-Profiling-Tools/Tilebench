import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _matmul_kernel(
    a,
    b,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    num_k_tiles = ct.cdiv(a.shape[1], TILE_K)

    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

    for k_tile in range(0, num_k_tiles):
        a_tile = ct.load(
            a,
            index=(bid_m, k_tile),
            shape=(TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(k_tile, bid_n),
            shape=(TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(output, index=(bid_m, bid_n), tile=ct.astype(acc, a.dtype))


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stream = torch.cuda.current_stream()

    TILE_M = 128
    TILE_N = 128
    TILE_K = 64
    occupancy = 8

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
    kernel = _matmul_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (a, b, output, TILE_M, TILE_N, TILE_K))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": TILE_M,
            "TILE_N": TILE_N,
            "TILE_K": TILE_K,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
