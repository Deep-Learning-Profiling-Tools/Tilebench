import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=2, num_worker_warps=8)
def _softmax(x, out):
    row = ct.bid(0)

    # Hold the complete row on chip without power-of-two tail padding.
    head = ct.load(
        x, (row, 0), (1, 8192), allow_tma=False
    ).astype(ct.float32)
    tail = ct.load(
        x, (row, 4), (1, 2048), allow_tma=False
    ).astype(ct.float32)

    # With the whole row retained, evaluate its private statistics directly.
    maximum = ct.maximum(
        ct.max(head, axis=1, keepdims=True),
        ct.max(tail, axis=1, keepdims=True),
    )
    head_exp = ct.exp2((head - maximum) * 1.4426950408889634)
    tail_exp = ct.exp2((tail - maximum) * 1.4426950408889634)
    denominator = (
        ct.sum(head_exp, axis=1, keepdims=True)
        + ct.sum(tail_exp, axis=1, keepdims=True)
    )
    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )

    ct.store(
        out, (row, 0), (head_exp * reciprocal).astype(out.dtype),
        allow_tma=False,
    )
    ct.store(
        out, (row, 4), (tail_exp * reciprocal).astype(out.dtype),
        allow_tma=False,
    )


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
        "head_cols": 8192,
        "tail_cols": 2048,
        "tail_tile_index": 4,
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 2,
        "num_worker_warps": 8,
        "allow_tma": False,
        "hold_complete_row": True,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
