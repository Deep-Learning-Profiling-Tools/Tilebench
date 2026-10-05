import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 8192 + 2048 -> the whole row is held on chip as two
# power-of-two tiles (no padding, no wasted exponentials).
_TILE_A = 8192          # first chunk: columns [0, 8192)
_TILE_B = 2048          # second chunk: columns [8192, 10240)
_ROWS_PER_PROGRAM = 1

_CONFIG = {
    "TILE_A": _TILE_A,
    "TILE_B": _TILE_B,
    "ROWS_PER_PROGRAM": _ROWS_PER_PROGRAM,
    "exp_formulation": "exp2_log2e_scale",
    "normalise": "reciprocal_multiply",
    "row_held_on_chip": True,
    "grid": "n_rows",
}


@ct.kernel
def _softmax_row_kernel(x, y, TILE_A: ConstInt, TILE_B: ConstInt, B_IDX: ConstInt):
    row = ct.bid(0)

    # Single read of the row from memory: two in-bounds tiles held on chip.
    xa = ct.load(x, (row, 0), (1, TILE_A),
                 padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    xb = ct.load(x, (row, B_IDX), (1, TILE_B),
                 padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)

    # Row statistics in fp32 over the held values (exact row maximum).
    m = ct.maximum(ct.max(xa, axis=1, keepdims=True),
                   ct.max(xb, axis=1, keepdims=True))          # (1, 1)

    # exp(x - m) via exp2 with log2(e) scale, fp32.
    pa = ct.exp2((xa - m) * 1.4426950408889634)
    pb = ct.exp2((xb - m) * 1.4426950408889634)

    l = ct.sum(pa, axis=1, keepdims=True) + ct.sum(pb, axis=1, keepdims=True)  # (1, 1)
    inv_l = 1.0 / l

    # Normalise in fp32, single downcast at the store.
    ya = (pa * inv_l).astype(y.dtype)
    yb = (pb * inv_l).astype(y.dtype)
    ct.store(y, (row, 0), ya)
    ct.store(y, (row, B_IDX), yb)


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols == _TILE_A + _TILE_B, "kernel is specialised for n_cols = 10240"
    assert _TILE_A % _TILE_B == 0
    y = torch.empty_like(x)
    grid = (n_rows // _ROWS_PER_PROGRAM,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _softmax_row_kernel,
        (x, y, _TILE_A, _TILE_B, _TILE_A // _TILE_B),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
