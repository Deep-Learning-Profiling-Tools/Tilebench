import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4)
def _softmax(x, out):
    row = ct.bid(0)

    # Keep the complete row on chip in five equally sized tiles.
    a = ct.load(x, (row, 0), (1, 2048), allow_tma=False).astype(ct.float32)
    b = ct.load(x, (row, 1), (1, 2048), allow_tma=False).astype(ct.float32)
    c = ct.load(x, (row, 2), (1, 2048), allow_tma=False).astype(ct.float32)
    d = ct.load(x, (row, 3), (1, 2048), allow_tma=False).astype(ct.float32)
    e = ct.load(x, (row, 4), (1, 2048), allow_tma=False).astype(ct.float32)

    column_max = ct.maximum(
        ct.maximum(ct.maximum(a, b), ct.maximum(c, d)), e
    )
    maximum = ct.max(column_max, axis=1, keepdims=True)

    ea = ct.exp2((a - maximum) * 1.4426950408889634)
    eb = ct.exp2((b - maximum) * 1.4426950408889634)
    ec = ct.exp2((c - maximum) * 1.4426950408889634)
    ed = ct.exp2((d - maximum) * 1.4426950408889634)
    ee = ct.exp2((e - maximum) * 1.4426950408889634)

    column_sum = ((ea + eb) + (ec + ed)) + ee
    denominator = ct.sum(column_sum, axis=1, keepdims=True)
    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )

    ct.store(out, (row, 0), (ea * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 1), (eb * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 2), (ec * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 3), (ed * reciprocal).astype(out.dtype), allow_tma=False)
    ct.store(out, (row, 4), (ee * reciprocal).astype(out.dtype), allow_tma=False)


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
        "tiles_per_row": 5,
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 4,
        "allow_tma": False,
        "hold_complete_row": True,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
