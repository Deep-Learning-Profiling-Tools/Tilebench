import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=2, num_worker_warps=8)
def _softmax(x, out):
    row = ct.bid(0)

    # The complete row is held on chip as one padded chunk.
    values = ct.load(
        x,
        (row, 0),
        (1, 16384),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
    ).astype(ct.float32)

    # Single-chunk statistics: no global intermediate storage.
    maximum = ct.max(values, axis=1, keepdims=True)
    exponentials = ct.exp2((values - maximum) * 1.4426950408889634)
    denominator = ct.sum(exponentials, axis=1, keepdims=True)

    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )
    result = (exponentials * reciprocal).astype(out.dtype)
    ct.store(out, (row, 0), result, allow_tma=False)


def run(x):
    out = torch.empty_like(x)
    ct.launch(
        torch.cuda.current_stream(),
        (2048,),
        _softmax,
        (x, out),
    )
    return out


def get_last_config() -> dict:
    return {
        "n_rows": 2048,
        "n_cols": 10240,
        "rows_per_program": 1,
        "tile_cols": 16384,
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 2,
        "num_worker_warps": 8,
        "allow_tma": False,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
