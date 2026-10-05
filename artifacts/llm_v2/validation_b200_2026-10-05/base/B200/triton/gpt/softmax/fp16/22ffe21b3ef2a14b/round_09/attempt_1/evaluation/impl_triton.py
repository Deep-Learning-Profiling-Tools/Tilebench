import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_two_held_rows(
    X,
    Y,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(
        0, ROWS_PER_PROGRAM
    )
    cols = tl.arange(0, BLOCK_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    # Both complete rows remain on chip; each input element is read once.
    x0 = tl.load(X + offsets).to(tl.float32)
    x1 = tl.load(X + offsets + BLOCK_COLS).to(tl.float32)
    x2 = tl.load(X + offsets + 2 * BLOCK_COLS).to(tl.float32)
    x3 = tl.load(X + offsets + 3 * BLOCK_COLS).to(tl.float32)
    x4 = tl.load(X + offsets + 4 * BLOCK_COLS).to(tl.float32)

    maxima = tl.maximum(x0, x1)
    maxima = tl.maximum(maxima, x2)
    maxima = tl.maximum(maxima, x3)
    maxima = tl.maximum(maxima, x4)
    row_max = tl.max(maxima, axis=1)

    log2e: tl.constexpr = 1.4426950408889634
    negative_scaled_max = -row_max[:, None] * log2e
    e0 = tl.exp2(tl.fma(x0, log2e, negative_scaled_max))
    e1 = tl.exp2(tl.fma(x1, log2e, negative_scaled_max))
    e2 = tl.exp2(tl.fma(x2, log2e, negative_scaled_max))
    e3 = tl.exp2(tl.fma(x3, log2e, negative_scaled_max))
    e4 = tl.exp2(tl.fma(x4, log2e, negative_scaled_max))

    sums = e0 + e1
    sums = sums + e2
    sums = sums + e3
    sums = sums + e4
    row_sum = tl.sum(sums, axis=1, dtype=tl.float32)
    inv_sum = (1.0 / row_sum)[:, None]

    tl.store(Y + offsets, e0 * inv_sum)
    tl.store(Y + offsets + BLOCK_COLS, e1 * inv_sum)
    tl.store(Y + offsets + 2 * BLOCK_COLS, e2 * inv_sum)
    tl.store(Y + offsets + 3 * BLOCK_COLS, e3 * inv_sum)
    tl.store(Y + offsets + 4 * BLOCK_COLS, e4 * inv_sum)


def run(x):
    y = torch.empty_like(x)
    _softmax_two_held_rows[(1024,)](
        x,
        y,
        ROWS_PER_PROGRAM=2,
        BLOCK_COLS=2048,
        N_COLS=10240,
        num_warps=16,
        num_stages=1,
        num_ctas=1,
        maxnreg=64,
    )
    return y


def get_last_config() -> dict:
    return {
        "n_rows": 2048,
        "n_cols": 10240,
        "block_cols": 2048,
        "held_fragments": 5,
        "rows_per_program": 2,
        "fragment_reduction": "serial",
        "exponential": "exp2_fma",
        "log2e": 1.4426950408889634,
        "load_cache_modifier": "",
        "store_cache_modifier": "",
        "num_warps": 16,
        "num_stages": 1,
        "num_ctas": 1,
        "maxnreg": 64,
    }
