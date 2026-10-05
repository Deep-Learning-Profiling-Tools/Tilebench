import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 5 * 2048 -> the whole row is held on chip as five (1, 2048) chunks.
# The row is read from memory exactly once; statistics are evaluated over the held
# values (max first, then sum of exponentials), which the contract permits for
# on-chip rows. Exponentials are computed once per element and reused for the
# normalisation (fp32), with a single downcast at the store.
_TILE = 2048            # chunk width along the row (power of two)
_NUM_CHUNKS = 5         # chunks per row (10240 / 2048), unrolled in the kernel
_ROWS_PER_PROGRAM = 1   # one row per program
_OCCUPANCY = 4          # register budget for the held fp32 row (no spills)
_LOG2E = 1.4426950408889634

_CONFIG = {
    "TILE": _TILE,
    "NUM_CHUNKS": _NUM_CHUNKS,
    "ROWS_PER_PROGRAM": _ROWS_PER_PROGRAM,
    "occupancy": _OCCUPANCY,
    "exp_formulation": "exp2_log2e_scale",
    "normalise": "reciprocal_multiply",
    "structure": "whole_row_on_chip_single_read",
    "grid": "n_rows",
}


@ct.kernel(occupancy=_OCCUPANCY)
def _softmax_row_kernel(x, y, TILE: ConstInt):
    row = ct.bid(0)

    # Single read of the row: all five chunk loads issued up front.
    t0 = ct.load(x, (row, 0), (1, TILE), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    t1 = ct.load(x, (row, 1), (1, TILE), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    t2 = ct.load(x, (row, 2), (1, TILE), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    t3 = ct.load(x, (row, 3), (1, TILE), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    t4 = ct.load(x, (row, 4), (1, TILE), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)

    # Row statistics over the held values (fp32): exact max, then sum of exponentials.
    mx = ct.maximum(ct.maximum(ct.maximum(t0, t1), ct.maximum(t2, t3)), t4)
    m = ct.max(mx, axis=1, keepdims=True)                      # (1, 1) exact row max
    ms = m * _LOG2E

    p0 = ct.exp2(t0 * _LOG2E - ms)
    p1 = ct.exp2(t1 * _LOG2E - ms)
    p2 = ct.exp2(t2 * _LOG2E - ms)
    p3 = ct.exp2(t3 * _LOG2E - ms)
    p4 = ct.exp2(t4 * _LOG2E - ms)

    s = (p0 + p1) + (p2 + p3) + p4
    l = ct.sum(s, axis=1, keepdims=True)                       # (1, 1) fp32 denominator
    inv_l = 1.0 / l

    # Normalise in fp32, single downcast at the store.
    ct.store(y, (row, 0), (p0 * inv_l).astype(y.dtype))
    ct.store(y, (row, 1), (p1 * inv_l).astype(y.dtype))
    ct.store(y, (row, 2), (p2 * inv_l).astype(y.dtype))
    ct.store(y, (row, 3), (p3 * inv_l).astype(y.dtype))
    ct.store(y, (row, 4), (p4 * inv_l).astype(y.dtype))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols == _NUM_CHUNKS * _TILE, "kernel is specialised for n_cols == 5 * 2048"
    y = torch.empty_like(x)
    grid = (n_rows // _ROWS_PER_PROGRAM,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_row_kernel,
        (x, y, _TILE),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
