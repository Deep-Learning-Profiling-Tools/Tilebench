import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _layernorm_kernel(x, weight, bias, output, eps, K: ConstInt, TILE: ConstInt):
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

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        wj = ct.load(
            weight,
            index=(j,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        bj = ct.load(
            bias,
            index=(j,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )

        xj_f32 = ct.astype(xj, np.float32)
        wj_f32 = ct.reshape(ct.astype(wj, np.float32), (1, TILE))
        bj_f32 = ct.reshape(ct.astype(bj, np.float32), (1, TILE))

        y = (xj_f32 - mean) * rstd * wj_f32 + bj_f32
        ct.store(
            output,
            index=(row, j),
            tile=ct.astype(y, x.dtype),
            allow_tma=False,
        )


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)

    K = x.shape[-1]
    n_rows = x.numel() // K
    x2d = x.reshape(n_rows, K)
    out2d = output.reshape(n_rows, K)

    TILE = 1024
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)
    ct.launch(stream, grid, _layernorm_kernel, (x2d, weight, bias, out2d, eps, K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
