import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _argmax_pair_combine(v1, i1, v2, i2):
    take_v2 = (v2 > v1) | ((v2 == v1) & (i2 < i1))
    return ct.where(take_v2, v2, v1), ct.where(take_v2, i2, i1)


@ct.kernel
def _argmax_rows_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)

    best_val = ct.full((1,), -np.inf, dtype=np.float32)
    best_idx = ct.full((1,), 0, dtype=np.int32)

    num_full_tiles = N // TILE

    for j in range(0, num_full_tiles):
        vals = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            latency=1,
            allow_tma=False,
        )

        idxs = offs[None, :] + j * TILE

        tile_val, tile_idx = ct.reduce(
            (vals, idxs),
            axis=1,
            func=_argmax_pair_combine,
            identity=(-np.inf, 2147483647),
            keepdims=False,
        )
        tile_val_f = ct.astype(tile_val, np.float32)

        take = tile_val_f > best_val
        best_val = ct.where(take, tile_val_f, best_val)
        best_idx = ct.where(take, tile_idx, best_idx)

    if N % TILE != 0:
        j = num_full_tiles
        vals = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )

        idxs = offs[None, :] + j * TILE
        valid = idxs < N
        idxs = ct.where(valid, idxs, 2147483647)

        tile_val, tile_idx = ct.reduce(
            (vals, idxs),
            axis=1,
            func=_argmax_pair_combine,
            identity=(-np.inf, 2147483647),
            keepdims=False,
        )
        tile_val_f = ct.astype(tile_val, np.float32)

        take = tile_val_f > best_val
        best_val = ct.where(take, tile_val_f, best_val)
        best_idx = ct.where(take, tile_idx, best_idx)

    ct.store(output, index=(row,), tile=ct.astype(best_idx, output.dtype))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim == -1:
        dim = 1
    if dim != 1:
        raise NotImplementedError("This TileBench argmax implementation supports row-wise dim=1 only.")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.int64)
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (M, 1, 1)
    kernel = _argmax_rows_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "FULL_TILE_FAST_PATH": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
