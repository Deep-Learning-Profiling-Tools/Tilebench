import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _mean_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.load(x, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.astype(xj, np.float32)
    s = ct.sum(acc, axis=1, keepdims=False)
    mean = s / float(N)
    ct.store(output, index=(row,), tile=mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4  # set via decorator

    grid = (M, 1, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
