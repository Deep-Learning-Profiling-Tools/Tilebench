```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _rmsnorm_kernel(x, rms_w, output, eps, K: ConstInt, TILE: ConstInt, ROWS: ConstInt):
    row_block = ct.bid(0)
    num_tiles = ct.cdiv(K, TILE)

    acc = ct.full((ROWS, TILE), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row_block, j),
            shape=(ROWS, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.astype(xj, np.float32)
        acc = acc + xj_f * xj_f

    sum_sq = ct.sum(acc, axis=1, keepdims=True)
    rstd = ct.rsqrt(sum_sq / K + eps)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row_block, j),
            shape=(ROWS, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.astype(xj, np.float32)

        wj = ct.load(
            rms_w,
            index=(j,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        wj_f = ct.astype(wj, np.float32)
        wj_f = ct.reshape(wj_f, (1, TILE))

        y = xj_f * rstd * wj_f
        ct.store(
            output,
            index=(row_block, j),
            tile=ct.astype(y, x.dtype),
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    n_rows = batch * M

    TILE = 1024
    ROWS = 2
    occupancy = 4

    x_2d = x.reshape(n_rows, K)
    output_2d = output.reshape(n_rows, K)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_rows, ROWS), 1, 1)
    kernel = _rmsnorm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_2d, rms_w, output_2d, eps, K, TILE, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "ROWS": ROWS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
