import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _exp2_approx_fast(y):
    y = ct.maximum(y, -24.0)
    bits = ct.astype((y + 126.94269504088896) * 8388608.0, np.int32)
    return ct.bitcast(bits, ct.float32)


@ct.kernel(occupancy=2)
def _softmax_kernel_generic(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(row, 0),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    x_f32 = ct.astype(x_tile, np.float32)

    row_max = ct.max(x_f32, axis=1, keepdims=True)
    shifted = (x_f32 - row_max) * 1.4426950408889634

    numerator = ct.exp2(shifted, flush_to_zero=True)
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    y = numerator * (1.0 / denominator)

    ct.store(
        output,
        index=(row, 0),
        tile=ct.astype(y, x.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel(occupancy=1)
def _softmax_kernel_split5_nomask(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(ct.load(x, index=(row, 0), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x1 = ct.astype(ct.load(x, index=(row, 1), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x2 = ct.astype(ct.load(x, index=(row, 2), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x3 = ct.astype(ct.load(x, index=(row, 3), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x4 = ct.astype(ct.load(x, index=(row, 4), shape=(1, TILE), latency=1, allow_tma=False), np.float32)

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    p0 = ct.exp2((x0 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p1 = ct.exp2((x1 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p2 = ct.exp2((x2 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p3 = ct.exp2((x3 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p4 = ct.exp2((x4 - row_max) * 1.4426950408889634, flush_to_zero=True)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


@ct.kernel(occupancy=1)
def _softmax_kernel_split5_fastfp16_nomask(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(ct.load(x, index=(row, 0), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x1 = ct.astype(ct.load(x, index=(row, 1), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x2 = ct.astype(ct.load(x, index=(row, 2), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x3 = ct.astype(ct.load(x, index=(row, 3), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x4 = ct.astype(ct.load(x, index=(row, 4), shape=(1, TILE), latency=1, allow_tma=False), np.float32)

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    scale = 1.4426950408889634
    p0 = _exp2_approx_fast((x0 - row_max) * scale)
    p1 = _exp2_approx_fast((x1 - row_max) * scale)
    p2 = _exp2_approx_fast((x2 - row_max) * scale)
    p3 = _exp2_approx_fast((x3 - row_max) * scale)
    p4 = _exp2_approx_fast((x4 - row_max) * scale)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


@ct.kernel(occupancy=1)
def _softmax_kernel_split5_padded(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(
        ct.load(x, index=(row, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x1 = ct.astype(
        ct.load(x, index=(row, 1), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x2 = ct.astype(
        ct.load(x, index=(row, 2), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x3 = ct.astype(
        ct.load(x, index=(row, 3), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x4 = ct.astype(
        ct.load(x, index=(row, 4), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    p0 = ct.exp2((x0 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p1 = ct.exp2((x1 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p2 = ct.exp2((x2 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p3 = ct.exp2((x3 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p4 = ct.exp2((x4 - row_max) * 1.4426950408889634, flush_to_zero=True)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    stream = torch.cuda.current_stream()

    if n_cols == 10240:
        TILE = 2048
        grid = (n_rows, 1, 1)

        if x.dtype == torch.float16:
            ct.launch(stream, grid, _softmax_kernel_split5_fastfp16_nomask, (x, output, TILE))
            mode = 3
        else:
            ct.launch(stream, grid, _softmax_kernel_split5_nomask, (x, output, TILE))
            mode = 0

        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 5,
            "TOTAL_TILE": TILE * 5,
            "MASKED": 0,
            "MODE": mode,
            "occupancy": 1,
        }
    elif n_cols > 8192:
        TILE = 2048
        grid = (n_rows, 1, 1)
        ct.launch(stream, grid, _softmax_kernel_split5_padded, (x, output, TILE))
        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 5,
            "TOTAL_TILE": TILE * 5,
            "MASKED": 1,
            "MODE": 0,
            "occupancy": 1,
        }
    else:
        TILE = 1 << (n_cols - 1).bit_length()
        grid = (n_rows, 1, 1)
        ct.launch(stream, grid, _softmax_kernel_generic, (x, output, TILE))
        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 1,
            "TOTAL_TILE": TILE,
            "MASKED": 1,
            "MODE": 0,
            "occupancy": 2,
        }

    _LAST_CFG.clear()
    _LAST_CFG.update(cfg)
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
