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
def _argmax_partials_nomask_kernel(x, tmp_vals, tmp_idx, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    chunk = ct.bid(1)

    offs = ct.arange(TILE, dtype=np.int32)
    vals = ct.load(
        x,
        index=(row, chunk),
        shape=(1, TILE),
        latency=1,
        allow_tma=False,
    )
    idxs = offs[None, :] + chunk * TILE

    tile_val, tile_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    ct.store(tmp_vals, index=(row, chunk), tile=tile_val, latency=1, allow_tma=False)
    ct.store(tmp_idx, index=(row, chunk), tile=tile_idx, latency=1, allow_tma=False)


@ct.kernel
def _argmax_partials_masked_kernel(x, tmp_vals, tmp_idx, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    chunk = ct.bid(1)

    offs = ct.arange(TILE, dtype=np.int32)
    vals = ct.load(
        x,
        index=(row, chunk),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )

    idxs = offs[None, :] + chunk * TILE
    valid = idxs < N
    idxs = ct.where(valid, idxs, 2147483647)

    tile_val, tile_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    ct.store(tmp_vals, index=(row, chunk), tile=tile_val, latency=1, allow_tma=False)
    ct.store(tmp_idx, index=(row, chunk), tile=tile_idx, latency=1, allow_tma=False)


@ct.kernel
def _argmax_finalize_kernel(tmp_vals, tmp_idx, output, NUM_CHUNKS: ConstInt, CHUNK_TILE: ConstInt):
    row = ct.bid(0)

    offs = ct.arange(CHUNK_TILE, dtype=np.int32)
    vals = ct.load(
        tmp_vals,
        index=(row, 0),
        shape=(1, CHUNK_TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    idxs = ct.load(
        tmp_idx,
        index=(row, 0),
        shape=(1, CHUNK_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    valid = offs[None, :] < NUM_CHUNKS
    idxs = ct.where(valid, idxs, 2147483647)

    _, best_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=False,
    )

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
    CHUNK_TILE = 8
    partial_occupancy = 4
    final_occupancy = 8

    num_chunks = ct.cdiv(N, TILE)
    full_tile_fast_path = (N % TILE) == 0

    tmp_vals = torch.empty((M, num_chunks), device=x.device, dtype=x.dtype)
    tmp_idx = torch.empty((M, num_chunks), device=x.device, dtype=torch.int32)

    partial_grid = (M, num_chunks, 1)
    if full_tile_fast_path:
        partial_kernel = _argmax_partials_nomask_kernel.with_hints(occupancy=partial_occupancy)
        ct.launch(stream, partial_grid, partial_kernel, (x, tmp_vals, tmp_idx, N, TILE))
    else:
        partial_kernel = _argmax_partials_masked_kernel.with_hints(occupancy=partial_occupancy)
        ct.launch(stream, partial_grid, partial_kernel, (x, tmp_vals, tmp_idx, N, TILE))

    final_kernel = _argmax_finalize_kernel.with_hints(occupancy=final_occupancy)
    ct.launch(stream, (M, 1, 1), final_kernel, (tmp_vals, tmp_idx, output, num_chunks, CHUNK_TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TWO_STAGE": True,
            "TILE": TILE,
            "CHUNK_TILE": CHUNK_TILE,
            "NUM_CHUNKS": num_chunks,
            "partial_occupancy": partial_occupancy,
            "final_occupancy": final_occupancy,
            "FULL_TILE_FAST_PATH": full_tile_fast_path,
            "CHUNK_TIE_FAST_PATH": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
