import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 5 * 2048 -> five full chunks per row, no tail masking needed.
# Two rows per program -> each chunk load moves 8 KB (2 x 2048 fp16), grid = 1024
# programs; occupancy 7 gives 148 * 7 = 1036 resident slots >= 1024 -> single wave.
_TILE = 2048            # chunk width along the row (power of two)
_ROWS_PER_PROGRAM = 2   # rows handled by one program (power of two)
_OCCUPANCY = 7          # expected resident CTAs per SM
_ALLOW_TMA = False      # register-path loads/stores (no smem staging)
_LOG2E = 1.4426950408889634

_CONFIG = {
    "TILE": _TILE,
    "ROWS_PER_PROGRAM": _ROWS_PER_PROGRAM,
    "occupancy": _OCCUPANCY,
    "allow_tma": _ALLOW_TMA,
    "exp_formulation": "exp2_log2e_scale",
    "normalise": "reciprocal_multiply",
    "structure": "online_two_sweep_chunked",
    "grid": "n_rows / ROWS_PER_PROGRAM",
}


@ct.kernel(occupancy=_OCCUPANCY)
def _softmax_online_kernel(x, y, NUM_CHUNKS: ConstInt, ROWS: ConstInt, TILE: ConstInt):
    rb = ct.bid(0)

    # Sweep 1: online (streaming, rescaling) row statistics in fp32, per row.
    m = ct.full((ROWS, 1), float("-inf"), ct.float32)
    l = ct.zeros((ROWS, 1), ct.float32)
    for c in range(NUM_CHUNKS):
        t = ct.load(x, (rb, c), (ROWS, TILE),
                    padding_mode=ct.PaddingMode.NEG_INF,
                    allow_tma=False).astype(ct.float32)
        m_c = ct.max(t, axis=1, keepdims=True)                 # (ROWS, 1)
        m_new = ct.maximum(m, m_c)
        alpha = ct.exp2((m - m_new) * _LOG2E)                  # rescale factor
        p = ct.exp2((t - m_new) * _LOG2E)
        l = l * alpha + ct.sum(p, axis=1, keepdims=True)
        m = m_new

    inv_l = 1.0 / l                                            # fp32, (ROWS, 1)
    m_scaled = m * _LOG2E

    # Sweep 2: normalise in fp32, single downcast at the store.
    for c in range(NUM_CHUNKS):
        t = ct.load(x, (rb, c), (ROWS, TILE),
                    padding_mode=ct.PaddingMode.NEG_INF,
                    allow_tma=False).astype(ct.float32)
        yv = ct.exp2(t * _LOG2E - m_scaled) * inv_l
        ct.store(y, (rb, c), yv.astype(y.dtype), allow_tma=False)


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols % _TILE == 0, "kernel is specialised for n_cols divisible by TILE"
    assert n_rows % _ROWS_PER_PROGRAM == 0, "kernel is specialised for n_rows divisible by ROWS_PER_PROGRAM"
    y = torch.empty_like(x)
    num_chunks = n_cols // _TILE
    grid = (n_rows // _ROWS_PER_PROGRAM,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_online_kernel,
        (x, y, num_chunks, _ROWS_PER_PROGRAM, _TILE),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
