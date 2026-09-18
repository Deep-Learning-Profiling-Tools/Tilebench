import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _mean_reduction_kernel(x, output, N: ConstInt, TILE: ConstInt, EVEN_N: ConstBool):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        if EVEN_N:
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        acc = acc + ct.astype(xj, np.float32)

    total = ct.sum(acc)
    mean = total / N
    ct.store(output, index=(row,), tile=mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    TILE = 1024
    occupancy = 4
    EVEN_N = (N % TILE) == 0

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    kernel = _mean_reduction_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, N, TILE, EVEN_N))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "EVEN_N": EVEN_N,
            "allow_tma": False,
            "latency": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
