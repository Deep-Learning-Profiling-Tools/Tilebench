import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_held_row(
    X,
    Y,
    BLOCK_COLS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    base = row * N_COLS

    # Five fragments comprise one complete row held on chip.
    # Every input element is read exactly once.
    x0 = tl.load(X + base + cols).to(tl.float32)
    x1 = tl.load(X + base + BLOCK_COLS + cols).to(tl.float32)
    x2 = tl.load(X + base + 2 * BLOCK_COLS + cols).to(tl.float32)
    x3 = tl.load(X + base + 3 * BLOCK_COLS + cols).to(tl.float32)
    x4 = tl.load(X + base + 4 * BLOCK_COLS + cols).to(tl.float32)

    max01 = tl.maximum(x0, x1)
    max23 = tl.maximum(x2, x3)
    row_max = tl.max(
        tl.maximum(tl.maximum(max01, max23), x4), axis=0
    )

    e0 = tl.exp(x0 - row_max)
    e1 = tl.exp(x1 - row_max)
    e2 = tl.exp(x2 - row_max)
    e3 = tl.exp(x3 - row_max)
    e4 = tl.exp(x4 - row_max)

    row_sum = tl.sum(
        ((e0 + e1) + (e2 + e3)) + e4,
        axis=0,
        dtype=tl.float32,
    )
    inv_sum = 1.0 / row_sum

    # Reuse the on-chip exponentials; downcast only at the stores.
    tl.store(Y + base + cols, e0 * inv_sum)
    tl.store(Y + base + BLOCK_COLS + cols, e1 * inv_sum)
    tl.store(Y + base + 2 * BLOCK_COLS + cols, e2 * inv_sum)
    tl.store(Y + base + 3 * BLOCK_COLS + cols, e3 * inv_sum)
    tl.store(Y + base + 4 * BLOCK_COLS + cols, e4 * inv_sum)


def run(x):
    y = torch.empty_like(x)
    _softmax_held_row[(2048,)](
        x,
        y,
        BLOCK_COLS=2048,
        N_COLS=10240,
        num_warps=8,
        num_stages=1,
        num_ctas=1,
    )
    return y


def get_last_config() -> dict:
    return {
        "block_cols": 2048,
        "held_fragments": 5,
        "rows_per_program": 1,
        "num_warps": 8,
        "num_stages": 1,
        "num_ctas": 1,
    }
