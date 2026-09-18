import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _mean_kernel(x, output, N: ConstInt,
                 BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    pid = ct.bid(0)
    num_tiles = ct.cdiv(N, BLOCK_N)
    acc = ct.full((BLOCK_M,), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.load(x, index=(pid, j), shape=(BLOCK_M, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
        xf = ct.astype(xj, np.float32)
        acc = acc + ct.sum(xf, axis=1, keepdims=False)
    mean = acc / float(N)
    ct.store(output, index=(pid,), tile=mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    BLOCK_M = 4
    BLOCK_N = 2048
    occupancy = 4

    grid = (ct.cdiv(M, BLOCK_M), 1, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                      "occupancy": occupancy, "strategy": "multi-row-2d"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
