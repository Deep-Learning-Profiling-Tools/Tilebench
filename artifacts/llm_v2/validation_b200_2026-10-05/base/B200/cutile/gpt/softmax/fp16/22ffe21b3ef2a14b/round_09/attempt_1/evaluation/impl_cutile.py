import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=5, opt_level=3)
def _softmax(x, out):
    row = ct.bid(0)
    base = ct.assume_divisible_by(row * 10240, 2048)

    head_offsets = base + ct.arange(8192, dtype=ct.int32)
    tail_offsets = base + 8192 + ct.arange(2048, dtype=ct.int32)

    x_raw = x.get_raw_memory()
    head = x_raw.load_offset(head_offsets).astype(ct.float32)
    tail = x_raw.load_offset(tail_offsets).astype(ct.float32)

    # The complete row stays on chip for both logical stages.
    column_max = ct.max(head.reshape((4, 2048)), axis=0)
    column_max = ct.maximum(column_max, tail)
    maximum = ct.max(column_max, axis=0)

    head_exp = ct.exp2((head - maximum) * 1.4426950408889634)
    tail_exp = ct.exp2((tail - maximum) * 1.4426950408889634)

    column_sum = ct.sum(head_exp.reshape((4, 2048)), axis=0)
    column_sum = column_sum + tail_exp
    denominator = ct.sum(column_sum, axis=0)
    reciprocal = ct.truediv(
        1.0, denominator, rounding_mode=ct.RoundingMode.APPROX
    )

    out_raw = out.get_raw_memory()
    out_raw.store_offset(
        head_offsets, (head_exp * reciprocal).astype(out.dtype)
    )
    out_raw.store_offset(
        tail_offsets, (tail_exp * reciprocal).astype(out.dtype)
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
        "tail_reduction_chunk_cols": 2048,
        "tail_reduction": "single_column_merge",
        "grid": (2048,),
        "num_ctas": 1,
        "occupancy": 5,
        "opt_level": 3,
        "addressing": "raw_element_offsets",
        "row_stride": 10240,
        "row_offset_divisibility": 2048,
        "hold_complete_row": True,
        "exp_base": 2,
        "log2_e": 1.4426950408889634,
        "normalization": "approximate_reciprocal_multiply",
    }
