import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 5 * 2048: the whole row is held on chip as five (1, 2048)
# sub-tiles (one logical chunk), read from global memory exactly once.
_TILE = 2048            # sub-tile width along the row (power of two)
_NUM_SUBTILES = 5       # sub-tiles per row (10240 / 2048)
_ROWS_PER_PROGRAM = 1
_OCCUPANCY = 4          # expected resident CTAs per SM (row held in registers)
_LOG2E = 1.4426950408889634

_CONFIG = {
    "TILE": _TILE,
    "NUM_SUBTILES": _NUM_SUBTILES,
    "ROWS_PER_PROGRAM": _ROWS_PER_PROGRAM,
    "occupancy": _OCCUPANCY,
    "exp_formulation": "exp2_log2e_scale_ftz",
    "normalise": "reciprocal_multiply",
    "structure": "row_held_on_chip_single_read",
    "grid": "n_rows",
}


@ct.kernel(occupancy=_OCCUPANCY)
def _softmax_rowheld_kernel(x, y, TILE: ConstInt):
    row = ct.bid(0)

    # Single read of the whole row (five independent loads, issued together),
    # each upcast to fp32 before use.
    t0 = ct.load(x, (row, 0), (1, TILE)).astype(ct.float32)
    t1 = ct.load(x, (row, 1), (1, TILE)).astype(ct.float32)
    t2 = ct.load(x, (row, 2), (1, TILE)).astype(ct.float32)
    t3 = ct.load(x, (row, 3), (1, TILE)).astype(ct.float32)
    t4 = ct.load(x, (row, 4), (1, TILE)).astype(ct.float32)

    # Row statistics in fp32 over the held row (one chunk = the whole row).
    # Initial online state: m = -inf, l = 0.
    m = ct.full((1, 1), float("-inf"), ct.float32)
    l = ct.zeros((1, 1), ct.float32)

    mm = ct.maximum(ct.maximum(ct.maximum(t0, t1), ct.maximum(t2, t3)), t4)
    m_c = ct.max(mm, axis=1, keepdims=True)                    # exact row max, (1, 1)
    m_new = ct.maximum(m, m_c)
    alpha = ct.exp2((m - m_new) * _LOG2E, flush_to_zero=True)  # = 0 from initial state
    ms = m_new * _LOG2E

    p0 = ct.exp2(t0 * _LOG2E - ms, flush_to_zero=True)
    p1 = ct.exp2(t1 * _LOG2E - ms, flush_to_zero=True)
    p2 = ct.exp2(t2 * _LOG2E - ms, flush_to_zero=True)
    p3 = ct.exp2(t3 * _LOG2E - ms, flush_to_zero=True)
    p4 = ct.exp2(t4 * _LOG2E - ms, flush_to_zero=True)

    s = (p0 + p1) + (p2 + p3) + p4
    l = l * alpha + ct.sum(s, axis=1, keepdims=True)
    m = m_new

    inv_l = 1.0 / l                                            # fp32, (1, 1)

    # Normalise: y = exp(x - m) / l in fp32, single downcast at the store.
    ct.store(y, (row, 0), (p0 * inv_l).astype(y.dtype))
    ct.store(y, (row, 1), (p1 * inv_l).astype(y.dtype))
    ct.store(y, (row, 2), (p2 * inv_l).astype(y.dtype))
    ct.store(y, (row, 3), (p3 * inv_l).astype(y.dtype))
    ct.store(y, (row, 4), (p4 * inv_l).astype(y.dtype))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols == _NUM_SUBTILES * _TILE, "kernel is specialised for n_cols = 5 * 2048"
    y = torch.empty_like(x)
    grid = (n_rows // _ROWS_PER_PROGRAM,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_rowheld_kernel,
        (x, y, _TILE),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
