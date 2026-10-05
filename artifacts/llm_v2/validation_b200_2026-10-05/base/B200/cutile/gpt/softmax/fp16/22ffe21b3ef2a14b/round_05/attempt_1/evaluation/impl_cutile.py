import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4)
def _softmax(x, out):
    row = ct.bid(0)

    # Hold the entire row on chip using two contiguous loads.
    head = ct.load(
        x, (row, 0), (1, 8192), allow_tma=False
    ).astype(ct.float32)
    tail = ct.load(
        x, (row, 4), (1, 2048), allow_tma=False
    ).astype(ct.float32)

    head_groups = head.reshape((4, 2048))
    tail_values = tail.reshape((2048,))
    column_max = ct.maximum(
        ct.max(head_groups, axis=0), tail_values
    )
    maximum = ct.max(column_max, axis=0)

    head_exp = ct.exp2((head - maximum) * 1.4426950408889634)
    tail_exp = ct.exp2((tail - maximum) * 1.4426950408889634)

    column_sum = (
        ct.sum(head_exp.reshape((4, 2048)), axis=0)
        + tail_exp.reshape((2048,))
    )
    denominator = ct.sum(column_sum, axis=0)
    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )

    ct.store(
        out,
        (row, 0),
        (head_exp * reciprocal).astype(out.dtype),
        allow_tma=False,
    )
    ct.store(
        out,
        (row, 4),
        (tail_exp * reciprocal).astype(out.dtype),
        allow_tma=False,
    )


def run(x):
    out = torch.empty_like(x)
    ct.launch(torch.cuda.current_stream(), (2048,), _softmax, (x, out))
    return out


def get_last_config() -> dict:
    return {
        "n_rows": 2048,
        "n_cols": 10240,
        "rows_per_program": 1,
        "load_tile_cols": (8192, 2048),
        "reduction_group_shape": (4, 2048),
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 4,
        "allow_tma": False,
        "hold_complete_row": True,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
