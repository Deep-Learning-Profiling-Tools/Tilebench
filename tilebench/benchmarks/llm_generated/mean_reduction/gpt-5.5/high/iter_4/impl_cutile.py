import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _mean_reduction_split_partial_kernel(
    x,
    partials,
    N: ConstInt,
    TILE: ConstInt,
    NUM_PARTS: ConstInt,
):
    row = ct.bid(0)
    part = ct.bid(1)

    num_tiles = ct.cdiv(N, TILE)
    tiles_per_part = ct.cdiv(num_tiles, NUM_PARTS)
    start_tile = part * tiles_per_part

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for j in range(0, tiles_per_part):
        tile_idx = start_tile + j
        xj = ct.load(
            x,
            index=(row, tile_idx),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        acc = acc + ct.astype(xj, np.float32)

    total = ct.sum(acc)
    ct.store(partials, index=(row, part), tile=total, latency=1, allow_tma=False)


@ct.kernel(occupancy=4)
def _mean_reduction_mma_partial_kernel(
    x,
    partials,
    N: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_R: ConstInt,
    NUM_PARTS: ConstInt,
):
    bid_m = ct.bid(0)
    part = ct.bid(1)

    num_tiles = ct.cdiv(N, BLOCK_K)
    tiles_per_part = ct.cdiv(num_tiles, NUM_PARTS)
    start_tile = part * tiles_per_part

    acc = ct.full((BLOCK_M, BLOCK_R), 0.0, dtype=np.float32)
    ones = ct.full((BLOCK_K, BLOCK_R), 1.0, dtype=x.dtype)

    for j in range(0, tiles_per_part):
        tile_idx = start_tile + j
        xj = ct.load(
            x,
            index=(bid_m, tile_idx),
            shape=(BLOCK_M, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        acc = ct.mma(xj, ones, acc)

    partial = ct.sum(acc, axis=1, keepdims=True) * (1.0 / BLOCK_R)
    ct.store(partials, index=(bid_m, part), tile=partial, latency=1, allow_tma=False)


@ct.kernel(occupancy=4)
def _mean_reduction_finalize_kernel(
    partials,
    output,
    N: ConstInt,
    NUM_PARTS: ConstInt,
):
    row = ct.bid(0)

    vals = ct.load(
        partials,
        index=(row, 0),
        shape=(1, NUM_PARTS),
        padding_mode=ct.PaddingMode.UNDETERMINED,
        latency=1,
        allow_tma=False,
    )
    total = ct.sum(vals)
    mean = total * (1.0 / N)
    ct.store(output, index=(row,), tile=mean, latency=1, allow_tma=False)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    NUM_PARTS = 8
    TILE = 1024

    BLOCK_M = 16
    BLOCK_K = 128
    BLOCK_R = 8

    occupancy = 4
    USE_MMA = x.dtype != torch.float32

    partials = torch.empty((M, NUM_PARTS), device=x.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    if USE_MMA:
        grid_partial = (ct.cdiv(M, BLOCK_M), NUM_PARTS, 1)
        ct.launch(
            stream,
            grid_partial,
            _mean_reduction_mma_partial_kernel,
            (x, partials, N, BLOCK_M, BLOCK_K, BLOCK_R, NUM_PARTS),
        )
    else:
        grid_partial = (M, NUM_PARTS, 1)
        ct.launch(
            stream,
            grid_partial,
            _mean_reduction_split_partial_kernel,
            (x, partials, N, TILE, NUM_PARTS),
        )

    ct.launch(
        stream,
        (M, 1, 1),
        _mean_reduction_finalize_kernel,
        (partials, output, N, NUM_PARTS),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "hybrid_mma_f16bf16_split_fp32",
            "USE_MMA": USE_MMA,
            "TILE": TILE,
            "NUM_PARTS": NUM_PARTS,
            "BLOCK_M": BLOCK_M,
            "BLOCK_K": BLOCK_K,
            "BLOCK_R": BLOCK_R,
            "occupancy": occupancy,
            "allow_tma": False,
            "latency": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
