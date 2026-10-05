import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 5 * 2048 -> five full chunks per row, no tail masking needed.
_TILE = 2048            # chunk width along the row (power of two)
_ROWS_PER_PROGRAM = 1
_OCCUPANCY = 8          # expected resident CTAs per SM (small chunk -> low register use)
_LOG2E = 1.4426950408889634

_CONFIG = {
    "TILE": _TILE,
    "ROWS_PER_PROGRAM": _ROWS_PER_PROGRAM,
    "occupancy": _OCCUPANCY,
    "exp_formulation": "exp2_log2e_scale",
    "normalise": "reciprocal_multiply",
    "structure": "online_two_sweep_chunked",
    "grid": "n_rows",
}


@ct.kernel(occupancy=_OCCUPANCY)
def _softmax_online_kernel(x, y, NUM_CHUNKS: ConstInt, TILE: ConstInt):
    row = ct.bid(0)

    # Sweep 1: online (streaming, rescaling) row statistics in fp32.
    m = ct.full((1, 1), float("-inf"), ct.float32)
    l = ct.zeros((1, 1), ct.float32)
    for c in range(NUM_CHUNKS):
        t = ct.load(x, (row, c), (1, TILE),
                    padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
        m_c = ct.max(t, axis=1, keepdims=True)                 # (1, 1)
        m_new = ct.maximum(m, m_c)
        alpha = ct.exp2((m - m_new) * _LOG2E)                  # rescale factor
        p = ct.exp2((t - m_new) * _LOG2E)
        l = l * alpha + ct.sum(p, axis=1, keepdims=True)
        m = m_new

    inv_l = 1.0 / l                                            # fp32
    m_scaled = m * _LOG2E

    # Sweep 2: normalise in fp32, single downcast at the store.
    for c in range(NUM_CHUNKS):
        t = ct.load(x, (row, c), (1, TILE),
                    padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
        yv = ct.exp2(t * _LOG2E - m_scaled) * inv_l
        ct.store(y, (row, c), yv.astype(y.dtype))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols % _TILE == 0, "kernel is specialised for n_cols divisible by TILE"
    y = torch.empty_like(x)
    num_chunks = n_cols // _TILE
    grid = (n_rows // _ROWS_PER_PROGRAM,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_online_kernel,
        (x, y, num_chunks, _TILE),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
