import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _layernorm_stats_kernel(
    x,
    partial_sum,
    partial_sumsq,
    TILE: ConstInt,
):
    row = ct.bid(0)
    part = ct.bid(1)

    xj = ct.load(
        x,
        index=(row, part),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    xj_f32 = ct.astype(xj, np.float32)

    s = ct.sum(xj_f32, axis=1, keepdims=True)
    ss = ct.sum(xj_f32 * xj_f32, axis=1, keepdims=True)

    ct.store(
        partial_sum,
        index=(row, part),
        tile=s,
        allow_tma=False,
    )
    ct.store(
        partial_sumsq,
        index=(row, part),
        tile=ss,
        allow_tma=False,
    )


@ct.kernel
def _layernorm_apply_kernel(
    x,
    weight,
    bias,
    partial_sum,
    partial_sumsq,
    output,
    eps,
    K: ConstInt,
    TILE: ConstInt,
    PARTS_TILE: ConstInt,
):
    row = ct.bid(0)
    part = ct.bid(1)

    ps = ct.load(
        partial_sum,
        index=(row, 0),
        shape=(1, PARTS_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    pss = ct.load(
        partial_sumsq,
        index=(row, 0),
        shape=(1, PARTS_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    inv_k = 1.0 / K
    mean = ct.sum(ps) * inv_k
    var = ct.sum(pss) * inv_k - mean * mean
    var = ct.maximum(var, 0.0)
    rstd = ct.rsqrt(var + eps)

    xj = ct.load(
        x,
        index=(row, part),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    wj = ct.load(
        weight,
        index=(part,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    bj = ct.load(
        bias,
        index=(part,),
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
        index=(row, part),
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
    num_tiles = (K + TILE - 1) // TILE
    PARTS_TILE = 1 << (num_tiles - 1).bit_length()

    partial_sum = torch.empty((n_rows, num_tiles), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((n_rows, num_tiles), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    grid = (n_rows, num_tiles, 1)

    stats_kernel = _layernorm_stats_kernel.with_hints(occupancy=occupancy)
    apply_kernel = _layernorm_apply_kernel.with_hints(occupancy=occupancy)

    ct.launch(
        stream,
        grid,
        stats_kernel,
        (x2d, partial_sum, partial_sumsq, TILE),
    )
    ct.launch(
        stream,
        grid,
        apply_kernel,
        (x2d, weight, bias, partial_sum, partial_sumsq, out2d, eps, K, TILE, PARTS_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "PARTS_TILE": PARTS_TILE,
        "occupancy": occupancy,
        "split_passes": 2,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
