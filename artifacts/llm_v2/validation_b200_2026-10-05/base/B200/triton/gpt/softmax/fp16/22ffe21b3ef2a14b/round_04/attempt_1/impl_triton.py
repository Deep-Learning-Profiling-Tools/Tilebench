import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_held_row(
    X,
    Y,
    N_COLS: tl.constexpr,
    PAIR_COLS: tl.constexpr,
    TAIL_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N_COLS
    pair_offsets = tl.arange(0, PAIR_COLS)
    tail_offsets = tl.arange(0, TAIL_COLS)

    # Hold the complete row in three contiguous fragments.
    x01 = tl.load(X + base + pair_offsets).to(tl.float32)
    x23 = tl.load(X + base + PAIR_COLS + pair_offsets).to(tl.float32)
    x4 = tl.load(X + base + 2 * PAIR_COLS + tail_offsets).to(tl.float32)

    x01 = tl.reshape(x01, (TAIL_COLS, 2))
    x23 = tl.reshape(x23, (TAIL_COLS, 2))

    pair_max = tl.maximum(tl.max(x01, axis=1), tl.max(x23, axis=1))
    row_max = tl.max(tl.maximum(pair_max, x4), axis=0)

    e01 = tl.exp(x01 - row_max)
    e23 = tl.exp(x23 - row_max)
    e4 = tl.exp(x4 - row_max)

    partial_sum = (
        tl.sum(e01, axis=1, dtype=tl.float32)
        + tl.sum(e23, axis=1, dtype=tl.float32)
        + e4
    )
    row_sum = tl.sum(partial_sum, axis=0, dtype=tl.float32)
    inv_sum = 1.0 / row_sum

    tl.store(
        Y + base + pair_offsets,
        tl.reshape(e01 * inv_sum, (PAIR_COLS,)),
    )
    tl.store(
        Y + base + PAIR_COLS + pair_offsets,
        tl.reshape(e23 * inv_sum, (PAIR_COLS,)),
    )
    tl.store(
        Y + base + 2 * PAIR_COLS + tail_offsets,
        e4 * inv_sum,
    )


def run(x):
    y = torch.empty_like(x)
    _softmax_held_row[(2048,)](
        x,
        y,
        N_COLS=10240,
        PAIR_COLS=4096,
        TAIL_COLS=2048,
        num_warps=16,
        num_stages=1,
        num_ctas=1,
    )
    return y


def get_last_config() -> dict:
    return {
        "pair_cols": 4096,
        "paired_fragments": 2,
        "tail_cols": 2048,
        "rows_per_program": 1,
        "num_warps": 16,
        "num_stages": 1,
        "num_ctas": 1,
    }
