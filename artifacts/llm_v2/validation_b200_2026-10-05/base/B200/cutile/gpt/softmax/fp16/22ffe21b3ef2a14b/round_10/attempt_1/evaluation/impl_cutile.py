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

    # The complete row remains on chip throughout both logical stages.
    head_max = ct.max(head.reshape((8, 1024)), axis=0)
    tail_max = ct.max(tail.reshape((2, 1024)), axis=0)
    maximum = ct.max(ct.maximum(head_max, tail_max), axis=0)

    # Distribute the log2(e) scale to enable fused multiply-add lowering.
    negative_max_log2 = maximum * -1.4426950408889634
    head_exp = ct.exp2(head * 1.4426950408889634 + negative_max_log2)
    tail_exp = ct.exp2(tail * 1.4426950408889634 + negative_max_log2)

    head_sum = ct.sum(head_exp.reshape((8, 1024)), axis=0)
    tail_sum = ct.sum(tail_exp.reshape((2, 1024)), axis=0)
    denominator = ct.sum(head_sum + tail_sum, axis=0)
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
        "reduction_group_shape": (8, 1024),
        "tail_reduction_group_shape": (2, 1024),
        "tail_reduction": "balanced_column_merge",
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
        "exponent_formulation": "scaled_input_minus_scaled_row_maximum",
        "normalization": "approximate_reciprocal_multiply",
    }
