import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _mean_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    j = ct.bid(1)
    xj = ct.load(x, index=(row, j), shape=(1, TILE),
                 padding_mode=ct.PaddingMode.ZERO)
    xj_f = ct.astype(xj, np.float32)
    s = ct.sum(xj_f)  # 0D scalar tile, reduces all elements
    partial_mean = s / np.float32(N)
    ct.atomic_add(output, (row,), partial_mean,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.zeros((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    num_chunks = (N + TILE - 1) // TILE
    grid = (M, num_chunks, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy,
                      "strategy": "split-k-atomic"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
