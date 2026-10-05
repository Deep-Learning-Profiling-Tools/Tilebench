import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals.
# n_cols = 10240 = 5 * 2048: each row is held on chip as five (1, 2048) sub-tiles
# (one logical chunk), read from global memory exactly once.
# Persistent grid (148 SMs x 3 programs); each program walks rows with stride
# num_programs and prefetches the next row (register double-buffering) before
# normalising/storing the current one.
_TILE = 2048            # sub-tile width along the row (power of two)
_NUM_SUBTILES = 5       # sub-tiles per row (10240 / 2048)
_OCCUPANCY = 3          # resident programs per SM
_NUM_SMS = 148          # B200 SM count (device context)
_PREFETCH_DEPTH = 1     # rows prefetched ahead per program
_LOG2E = 1.4426950408889634

_CONFIG = {
    "TILE": _TILE,
    "NUM_SUBTILES": _NUM_SUBTILES,
    "occupancy": _OCCUPANCY,
    "num_sms": _NUM_SMS,
    "grid": "min(n_rows, num_sms * occupancy)",
    "prefetch_depth": _PREFETCH_DEPTH,
    "row_schedule": "persistent_strided",
    "exp_formulation": "exp2_log2e_scale_ftz",
    "normalise": "reciprocal_multiply",
    "structure": "row_held_on_chip_single_read",
}


@ct.kernel(occupancy=_OCCUPANCY)
def _softmax_persistent_kernel(x, y, n_rows: int, TILE: ConstInt):
    pid = ct.bid(0)
    nprog = ct.num_blocks(0)
    # rows owned by this program: pid, pid + nprog, ... (pid < n_rows guaranteed)
    n_my = (n_rows - 1 - pid) // nprog + 1

    # Prologue: load the first owned row (fp16, upcast at use).
    a0 = ct.load(x, (pid, 0), (1, TILE))
    a1 = ct.load(x, (pid, 1), (1, TILE))
    a2 = ct.load(x, (pid, 2), (1, TILE))
    a3 = ct.load(x, (pid, 3), (1, TILE))
    a4 = ct.load(x, (pid, 4), (1, TILE))
    row = pid

    for it in range(n_my - 1):
        nxt = row + nprog
        # Prefetch the next owned row before working on the current one.
        b0 = ct.load(x, (nxt, 0), (1, TILE))
        b1 = ct.load(x, (nxt, 1), (1, TILE))
        b2 = ct.load(x, (nxt, 2), (1, TILE))
        b3 = ct.load(x, (nxt, 3), (1, TILE))
        b4 = ct.load(x, (nxt, 4), (1, TILE))

        # ---- current row: statistics in fp32 (single chunk = whole row) ----
        t0 = a0.astype(ct.float32)
        t1 = a1.astype(ct.float32)
        t2 = a2.astype(ct.float32)
        t3 = a3.astype(ct.float32)
        t4 = a4.astype(ct.float32)

        m_init = ct.full((1, 1), float("-inf"), ct.float32)
        l_init = ct.zeros((1, 1), ct.float32)

        mm = ct.maximum(ct.maximum(ct.maximum(t0, t1), ct.maximum(t2, t3)), t4)
        m_c = ct.max(mm, axis=1, keepdims=True)                    # exact row max
        m_new = ct.maximum(m_init, m_c)
        alpha = ct.exp2((m_init - m_new) * _LOG2E, flush_to_zero=True)  # 0 from init
        ms = m_new * _LOG2E

        p0 = ct.exp2(t0 * _LOG2E - ms, flush_to_zero=True)
        p1 = ct.exp2(t1 * _LOG2E - ms, flush_to_zero=True)
        p2 = ct.exp2(t2 * _LOG2E - ms, flush_to_zero=True)
        p3 = ct.exp2(t3 * _LOG2E - ms, flush_to_zero=True)
        p4 = ct.exp2(t4 * _LOG2E - ms, flush_to_zero=True)

        s = (p0 + p1) + (p2 + p3) + p4
        l_new = l_init * alpha + ct.sum(s, axis=1, keepdims=True)
        inv_l = 1.0 / l_new

        # ---- normalise: y = exp(x - m) / l in fp32, single downcast ----
        ct.store(y, (row, 0), (p0 * inv_l).astype(y.dtype))
        ct.store(y, (row, 1), (p1 * inv_l).astype(y.dtype))
        ct.store(y, (row, 2), (p2 * inv_l).astype(y.dtype))
        ct.store(y, (row, 3), (p3 * inv_l).astype(y.dtype))
        ct.store(y, (row, 4), (p4 * inv_l).astype(y.dtype))

        a0 = b0
        a1 = b1
        a2 = b2
        a3 = b3
        a4 = b4
        row = nxt

    # Epilogue: last owned row (no prefetch).
    u0 = a0.astype(ct.float32)
    u1 = a1.astype(ct.float32)
    u2 = a2.astype(ct.float32)
    u3 = a3.astype(ct.float32)
    u4 = a4.astype(ct.float32)

    me_init = ct.full((1, 1), float("-inf"), ct.float32)
    le_init = ct.zeros((1, 1), ct.float32)

    mme = ct.maximum(ct.maximum(ct.maximum(u0, u1), ct.maximum(u2, u3)), u4)
    me_c = ct.max(mme, axis=1, keepdims=True)
    me_new = ct.maximum(me_init, me_c)
    alpha_e = ct.exp2((me_init - me_new) * _LOG2E, flush_to_zero=True)
    mse = me_new * _LOG2E

    q0 = ct.exp2(u0 * _LOG2E - mse, flush_to_zero=True)
    q1 = ct.exp2(u1 * _LOG2E - mse, flush_to_zero=True)
    q2 = ct.exp2(u2 * _LOG2E - mse, flush_to_zero=True)
    q3 = ct.exp2(u3 * _LOG2E - mse, flush_to_zero=True)
    q4 = ct.exp2(u4 * _LOG2E - mse, flush_to_zero=True)

    se = (q0 + q1) + (q2 + q3) + q4
    le_new = le_init * alpha_e + ct.sum(se, axis=1, keepdims=True)
    inv_le = 1.0 / le_new

    ct.store(y, (row, 0), (q0 * inv_le).astype(y.dtype))
    ct.store(y, (row, 1), (q1 * inv_le).astype(y.dtype))
    ct.store(y, (row, 2), (q2 * inv_le).astype(y.dtype))
    ct.store(y, (row, 3), (q3 * inv_le).astype(y.dtype))
    ct.store(y, (row, 4), (q4 * inv_le).astype(y.dtype))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    n_rows, n_cols = x.shape
    assert n_cols == _NUM_SUBTILES * _TILE, "kernel is specialised for n_cols = 5 * 2048"
    y = torch.empty_like(x)
    num_programs = min(n_rows, _NUM_SMS * _OCCUPANCY)
    ct.launch(
        torch.cuda.current_stream(),
        (num_programs,),
        _softmax_persistent_kernel,
        (x, y, n_rows, _TILE),
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
