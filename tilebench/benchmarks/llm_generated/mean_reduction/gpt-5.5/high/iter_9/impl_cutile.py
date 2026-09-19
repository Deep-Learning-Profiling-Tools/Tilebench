import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _mean_reduction_pair_flat_kernel(x_flat, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    full_pairs = num_tiles // 2

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for p in range(0, full_pairs):
        tile0 = 2 * p
        tile1 = tile0 + 1

        x0 = ct.load(
            x_flat,
            index=(row * num_tiles + tile0,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        x1 = ct.load(
            x_flat,
            index=(row * num_tiles + tile1,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        acc = acc + ct.astype(x0, np.float32) + ct.astype(x1, np.float32)

    if num_tiles != full_pairs * 2:
        last_tile = full_pairs * 2
        x_last = ct.load(
            x_flat,
            index=(row * num_tiles + last_tile,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        acc = acc + ct.astype(x_last, np.float32)

    total = ct.sum(acc)
    mean = total * (1.0 / N)
    ct.store(output, index=(row,), tile=mean, latency=1, allow_tma=False)


@ct.kernel(occupancy=4)
def _mean_reduction_pair_2d_kernel(
    x,
    output,
    N: ConstInt,
    TILE: ConstInt,
    EVEN_N: ConstBool,
):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    full_pairs = num_tiles // 2

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for p in range(0, full_pairs):
        tile0 = 2 * p
        tile1 = tile0 + 1

        if EVEN_N:
            x0 = ct.load(
                x,
                index=(row, tile0),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
            x1 = ct.load(
                x,
                index=(row, tile1),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            x0 = ct.load(
                x,
                index=(row, tile0),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
            x1 = ct.load(
                x,
                index=(row, tile1),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )

        acc = acc + ct.astype(x0, np.float32) + ct.astype(x1, np.float32)

    if num_tiles != full_pairs * 2:
        last_tile = full_pairs * 2
        if EVEN_N:
            x_last = ct.load(
                x,
                index=(row, last_tile),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            x_last = ct.load(
                x,
                index=(row, last_tile),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        acc = acc + ct.astype(x_last, np.float32)

    total = ct.sum(acc)
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

    TILE = 2048
    occupancy = 4
    EVEN_N = (N % TILE) == 0
    USE_FLAT = EVEN_N and x.is_contiguous()

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    if USE_FLAT:
        x_flat = x.reshape((M * N,))
        ct.launch(
            stream,
            grid,
            _mean_reduction_pair_flat_kernel,
            (x_flat, output, N, TILE),
        )
    else:
        ct.launch(
            stream,
            grid,
            _mean_reduction_pair_2d_kernel,
            (x, output, N, TILE, EVEN_N),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "flat_pair_unroll2" if USE_FLAT else "2d_pair_unroll2",
            "TILE": TILE,
            "PAIR_UNROLL": 2,
            "occupancy": occupancy,
            "EVEN_N": EVEN_N,
            "USE_FLAT": USE_FLAT,
            "allow_tma": False,
            "latency": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
