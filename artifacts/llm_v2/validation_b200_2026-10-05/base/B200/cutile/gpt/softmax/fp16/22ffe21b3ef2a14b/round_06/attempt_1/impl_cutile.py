import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _softmax(x, out):
    row = ct.bid(0)

    # Keep the complete row resident in five equally sized tiles.
    x0 = ct.load(x, (row, 0), (1, 2048), allow_tma=False).astype(ct.float32)
    x1 = ct.load(x, (row, 1), (1, 2048), allow_tma=False).astype(ct.float32)
    x2 = ct.load(x, (row, 2), (1, 2048), allow_tma=False).astype(ct.float32)
    x3 = ct.load(x, (row, 3), (1, 2048), allow_tma=False).astype(ct.float32)
    x4 = ct.load(x, (row, 4), (1, 2048), allow_tma=False).astype(ct.float32)

    column_max = ct.maximum(x0, x1)
    column_max = ct.maximum(column_max, x2)
    column_max = ct.maximum(column_max, x3)
    column_max = ct.maximum(column_max, x4)
    maximum = ct.max(column_max)

    e0 = ct.exp2((x0 - maximum) * 1.4426950408889634)
    e1 = ct.exp2((x1 - maximum) * 1.4426950408889634)
    e2 = ct.exp2((x2 - maximum) * 1.4426950408889634)
    e3 = ct.exp2((x3 - maximum) * 1.4426950408889634)
    e4 = ct.exp2((x4 - maximum) * 1.4426950408889634)

    column_sum = e0 + e1
    column_sum = column_sum + e2
    column_sum = column_sum + e3
    column_sum = column_sum + e4
    denominator = ct.sum(column_sum)
    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )

    ct.store(out, (row, 0), (e0 * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 1), (e1 * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 2), (e2 * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 3), (e3 * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 4), (e4 * reciprocal).astype(out.dtype), allow_tma=False)


def run(x):
    out = torch.empty_like(x)
    ct.launch(torch.cuda.current_stream(), (2048,), _softmax, (x, out))
    return out


def get_last_config() -> dict:
    return {
        "n_rows": 2048,
        "n_cols": 10240,
        "rows_per_program": 1,
        "tile_cols": 2048,
        "held_tiles": 5,
        "reduction": "elementwise_columns_then_row",
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 4,
        "opt_level": 3,
        "allow_tma": False,
        "hold_complete_row": True,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
