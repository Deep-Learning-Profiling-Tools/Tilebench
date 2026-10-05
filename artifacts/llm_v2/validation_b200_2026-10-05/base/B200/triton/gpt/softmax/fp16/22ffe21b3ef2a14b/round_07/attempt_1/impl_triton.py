import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_persistent(
    X,
    Y,
    BLOCK_COLS: tl.constexpr,
    N_COLS: tl.constexpr,
    N_ROWS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_COLS)
    log2e: tl.constexpr = 1.4426950408889634

    for row in tl.range(
        tl.program_id(0),
        N_ROWS,
        tl.num_programs(0),
        num_stages=LOOP_STAGES,
    ):
        base = row * N_COLS

        # Hold the entire row on chip; each input element is read once.
        x0 = tl.load(X + base + cols).to(tl.float32)
        x1 = tl.load(X + base + BLOCK_COLS + cols).to(tl.float32)
        x2 = tl.load(X + base + 2 * BLOCK_COLS + cols).to(tl.float32)
        x3 = tl.load(X + base + 3 * BLOCK_COLS + cols).to(tl.float32)
        x4 = tl.load(X + base + 4 * BLOCK_COLS + cols).to(tl.float32)

        maxima = tl.maximum(x0, x1)
        maxima = tl.maximum(maxima, x2)
        maxima = tl.maximum(maxima, x3)
        maxima = tl.maximum(maxima, x4)
        row_max = tl.max(maxima, axis=0)

        negative_scaled_max = -row_max * log2e
        e0 = tl.exp2(tl.fma(x0, log2e, negative_scaled_max))
        e1 = tl.exp2(tl.fma(x1, log2e, negative_scaled_max))
        e2 = tl.exp2(tl.fma(x2, log2e, negative_scaled_max))
        e3 = tl.exp2(tl.fma(x3, log2e, negative_scaled_max))
        e4 = tl.exp2(tl.fma(x4, log2e, negative_scaled_max))

        sums = e0 + e1
        sums = sums + e2
        sums = sums + e3
        sums = sums + e4
        row_sum = tl.sum(sums, axis=0, dtype=tl.float32)
        inv_sum = 1.0 / row_sum

        tl.store(Y + base + cols, e0 * inv_sum)
        tl.store(Y + base + BLOCK_COLS + cols, e1 * inv_sum)
        tl.store(Y + base + 2 * BLOCK_COLS + cols, e2 * inv_sum)
        tl.store(Y + base + 3 * BLOCK_COLS + cols, e3 * inv_sum)
        tl.store(Y + base + 4 * BLOCK_COLS + cols, e4 * inv_sum)


def run(x):
    y = torch.empty_like(x)
    _softmax_persistent[(592,)](
        x,
        y,
        BLOCK_COLS=2048,
        N_COLS=10240,
        N_ROWS=2048,
        LOOP_STAGES=2,
        num_warps=8,
        num_stages=1,
        num_ctas=1,
    )
    return y


def get_last_config() -> dict:
    return {
        "n_rows": 2048,
        "n_cols": 10240,
        "block_cols": 2048,
        "held_fragments": 5,
        "rows_per_iteration": 1,
        "persistent_programs": 592,
        "row_schedule": "grid_stride",
        "loop_stages": 2,
        "fragment_reduction": "serial",
        "exponential": "exp2_fma",
        "log2e": 1.4426950408889634,
        "num_warps": 8,
        "num_stages": 1,
        "num_ctas": 1,
    }
