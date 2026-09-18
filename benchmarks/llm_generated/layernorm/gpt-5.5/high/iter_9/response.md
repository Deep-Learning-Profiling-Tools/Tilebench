```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _layernorm_stats_kernel(
    x,
    mean_out,
    rstd_out,
    eps,
    K: ConstInt,
    TILE: ConstInt,
):
    row = ct.bid(0)
    num_tiles = ct.cdiv(K, TILE)

    sum_acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    sumsq_acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f32 = ct.astype(xj, np.float32)
        sum_acc = sum_acc + xj_f32
        sumsq_acc = sumsq_acc + xj_f32 * xj_f32

    inv_k = 1.0 / K
    mean = ct.sum(sum_acc) * inv_k
    var = ct.sum(sumsq_acc) * inv_k - mean * mean
    var = ct.maximum(var, 0.0)
    rstd = ct.rsqrt(var + eps)

    ct.store(mean_out, index=(row,), tile=mean, allow_tma=False)
    ct.store(rstd_out, index=(row,), tile=rstd, allow_tma=False)


@ct.kernel(occupancy=4)
def _layernorm_norm_kernel(
    x,
    weight,
    bias,
    mean,
    rstd,
    output,
    TILE: ConstInt,
    ROWS: ConstInt,
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    xj = ct.load(
        x,
        index=(row_block, col_block),
        shape=(ROWS, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    wj = ct.load(
        weight,
        index=(col_block,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    bj = ct.load(
        bias,
        index=(col_block,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    mj = ct.load(
        mean,
        index=(row_block,),
        shape=(ROWS,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    rj = ct.load(
        rstd,
        index=(row_block,),
        shape=(ROWS,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    xj_f32 = ct.astype(xj, np.float32)
    wj_f32 = ct.reshape(ct.astype(wj, np.float32), (1, TILE))
    bj_f32 = ct.reshape(ct.astype(bj, np.float32), (1, TILE))
    mean_f32 = ct.reshape(ct.astype(mj, np.float32), (ROWS, 1))
    rstd_f32 = ct.reshape(ct.astype(rj, np.float32), (ROWS, 1))

    y = (xj_f32 - mean_f32) * rstd_f32 * wj_f32 + bj_f32

    ct.store(
        output,
        index=(row_block, col_block),
        tile=ct.astype(y, x.dtype),
        allow_tma=False,
    )


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)

    K = x.shape[-1]
    n_rows = x.numel() // K
    x2d = x.reshape(n_rows, K)
    out2d = output.reshape(n_rows, K)

    mean = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    TILE_STATS = 1024
    TILE_NORM = 256
    ROWS = 8
    occupancy_stats = 4
    occupancy_norm = 4

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (n_rows, 1, 1),
        _layernorm_stats_kernel,
        (x2d, mean, rstd, eps, K, TILE_STATS),
    )

    ct.launch(
        stream,
        (ct.cdiv(n_rows, ROWS), ct.cdiv(K, TILE_NORM), 1),
        _layernorm_norm_kernel,
        (x2d, weight, bias, mean, rstd, out2d, TILE_NORM, ROWS),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_STATS": TILE_STATS,
            "TILE_NORM": TILE_NORM,
            "ROWS": ROWS,
            "occupancy_stats": occupancy_stats,
            "occupancy_norm": occupancy_norm,
            "split": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
