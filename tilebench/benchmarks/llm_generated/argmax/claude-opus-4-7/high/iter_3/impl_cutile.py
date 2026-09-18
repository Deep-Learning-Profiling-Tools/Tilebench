import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _argmax_kernel(x, output, N: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)

    best_val = ct.full((1,), -np.inf, dtype=np.float32)
    best_idx = ct.full((1,), 0, dtype=np.int64)

    num_tiles = ct.cdiv(N, TILE_N)
    for j in range(num_tiles):
        tile2d = ct.load(
            x, index=(row, j), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile2d.reshape((TILE_N,)), np.float32)

        local_max = ct.max(tile, axis=0, keepdims=True)
        local_arg = ct.argmax(tile, axis=0, keepdims=True)
        global_idx = ct.astype(local_arg, np.int64) + j * TILE_N

        new_better = local_max > best_val
        best_val = ct.where(new_better, local_max, best_val)
        best_idx = ct.where(new_better, global_idx, best_idx)

    ct.store(output, index=(row,), tile=best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    TILE_N = 4096

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _argmax_kernel, (x, output, N, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": 8})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
