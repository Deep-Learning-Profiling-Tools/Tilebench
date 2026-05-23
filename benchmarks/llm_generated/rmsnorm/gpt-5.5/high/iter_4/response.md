```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _rmsnorm_kernel(x, rms_w, output, eps, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(K, TILE)

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.reshape(ct.astype(xj, np.float32), (TILE,))
        acc = acc + xj_f * xj_f

    sum_sq = ct.sum(acc)
    rstd = ct.rsqrt(sum_sq / K + eps)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.reshape(ct.astype(xj, np.float32), (TILE,))

        wj = ct.load(
            rms_w,
            index=(j,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        wj_f = ct.astype(wj, np.float32)

        y = xj_f * rstd * wj_f
        y_out = ct.reshape(ct.astype(y, x.dtype), (1, TILE))

        ct.store(
            output,
            index=(row, j),
            tile=y_out,
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
    occupancy = 8

    x_2d = x.reshape(n_rows, K)
    output_2d = output.reshape(n_rows, K)

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)
    kernel = _rmsnorm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_2d, rms_w, output_2d, eps, K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
