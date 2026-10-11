"""
moe_topk_gating for NKI 0.7.0b1 on Trn2 (LNC=2: one logical core = 2 physical cores).

Mapping
  * Each program (physical core) owns M/2 rows.
  * Partition p of program `pid` owns Q = M/256 consecutive rows.
  * Rows are processed in tiles of R rows per partition (one [128, R*E] SBUF block).
    R is a fixed function of Q (4 if Q <= 16, else 8).

Per tile (stages 2-5 fused, nothing goes to HBM in between)
  * Row load: one DMA of the fp16 logits (a single traversal of the row), then an exact
    upcast to float32 on ScalarE.
  * Fused max+argmax encoding:
    - The fp16 -> fp32 upcast leaves the low 13 mantissa bits zero.
    - The column index (7 bits for E=128) is OR-ed into those bits with exact int32 math
      on GpSimd.
    - Distinct values keep their float32 order, so a float32 max-reduction returns the
      exact maximum (high bits) together with its column (low bits).
  * k sequential rounds:
    - Each round takes a full-row fp32 max (VectorE tensor_reduce over the E axis) and
      records it in slot j.
    - The selected column is then excluded by overwriting it with -inf:
      v = (row >= max) * (-FLT_MAX), stored as bf16 (rounds to -inf; other columns get
      -0), then row = row + v.
    - Every other column keeps its exact bits.
  * The tiles are software-pipelined over three stages:
    - A: load, upcast, tag.
    - B: round-0 max and mask.
    - C: exclusion and the remaining rounds.

Epilogue (on chip)
  * Decode: index = bits & mask; value = bits - index (exact fp32 value).
  * Softmax over the k selected values only, in fp32: subtract the max, exp, sum, divide.
  * Weights are interleaved in fp32 (4-byte aligned), then cast to fp16 with one
    contiguous copy. Indices are int32.
  * One DMA per output per program.
"""
import torch
import nki
import nki.language as nl
import nki.isa as nisa

try:  # registers the "neuron" device if not yet registered (host side only)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

PART = 128
LNC = 2
NEG_BIG = -3.4028234663852886e38     # -FLT_MAX; becomes -inf when rounded to bf16

_LAST_CONFIG = {}


def _rows_of_tile(t, n_full, r_tile, r_rem):
    R = r_tile
    if t >= n_full:
        R = r_rem
    return R


def _load_encode(x2, iot, p0, q0, R, E, in_dtype):
    # one traversal of the rows -> exact fp32 upcast -> tag low mantissa bits with column
    F = R * E
    xh = nl.ndarray((PART, F), dtype=in_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xh, src=x2[p0:p0 + PART, q0 * E:q0 * E + F])
    xf = nl.ndarray((PART, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xf, src=xh, engine=nisa.engine.scalar)
    xi = xf.view(nl.int32)
    nisa.tensor_tensor(dst=xi, data1=xi, data2=iot[:, 0:F], op=nl.bitwise_or,
                       engine=nisa.engine.gpsimd)
    return xf


def _round_max(xf, sel, col0, R, E):
    # full-row fp32 max (value + column via tag bits) of R rows -> sel[:, col0:col0+R]
    nisa.tensor_reduce(dst=sel[:, col0:col0 + R], op=nl.maximum,
                       data=xf.reshape((PART, R, E)), axis=2)


def _round_mask(xf, sel, col0, R, E):
    # v = (row >= selected max) * (-FLT_MAX) -> bf16: -inf at the selected column, -0 elsewhere
    vb = nl.ndarray((PART, R * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    for r in range(R):
        c = col0 + r
        nisa.tensor_scalar(dst=vb[:, r * E:(r + 1) * E], data=xf[:, r * E:(r + 1) * E],
                           op0=nl.greater_equal, operand0=sel[:, c:c + 1],
                           op1=nl.multiply, operand1=NEG_BIG,
                           engine=nisa.engine.vector)
    return vb


def _round_exclude(xf, vb, on_pool):
    # overwrite the selected column with -inf (others: x + (-0) == x exactly)
    if on_pool:
        nisa.tensor_tensor(dst=xf, data1=xf, data2=vb, op=nl.add,
                           engine=nisa.engine.gpsimd)
    else:
        nisa.tensor_tensor(dst=xf, data1=xf, data2=vb, op=nl.add,
                           engine=nisa.engine.vector)


def _rounds_after_first_exclusion(xf, sel, q0, R, E, Q, k, on_pool):
    # round 1 max (the first exclusion was already applied), then rounds 2..k-1
    _round_max(xf, sel, Q + q0, R, E)
    for j in range(2, k):
        vb = _round_mask(xf, sel, (j - 1) * Q + q0, R, E)
        _round_exclude(xf, vb, on_pool)
        _round_max(xf, sel, j * Q + q0, R, E)


@nki.jit
def _moe_topk_gating_kernel(logits, k, r_tile, idx_mask):
    M, E = logits.shape
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    Q = M // (n_prog * PART)        # rows per partition handled by this program
    QK = Q * k

    weights = nl.ndarray((M, k), dtype=logits.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray((M, k), dtype=nl.int32, buffer=nl.shared_hbm)

    x2 = logits.reshape((n_prog * PART, Q * E))
    w2 = weights.reshape((n_prog * PART, QK))
    i2 = indices.reshape((n_prog * PART, QK))
    p0 = pid * PART

    n_full = Q // r_tile
    r_rem = Q - n_full * r_tile
    n_tiles = n_full
    if r_rem > 0:
        n_tiles = n_full + 1
    r_alloc = r_tile
    if n_full == 0:
        r_alloc = r_rem

    # column index pattern 0..E-1 repeated for each row slot of a tile
    iot = nl.ndarray((PART, r_alloc * E), dtype=nl.int32, buffer=nl.sbuf)
    for r in range(r_alloc):
        nisa.iota(dst=iot[:, r * E:(r + 1) * E], pattern=[[1, E]], offset=0,
                  channel_multiplier=0)

    # selected (tagged) values, slot-major: column j*Q + q = slot j of local row q
    sel = nl.ndarray((PART, QK), dtype=nl.float32, buffer=nl.sbuf)

    # 3-stage software pipeline: A(s) load/encode, B(s-1) round-0, C(s-2) remaining rounds
    xf_b = sel
    xf_c = sel
    vb_c = sel
    for s in range(n_tiles + 2):
        tc = s - 2
        tb = s - 1
        # C1: first exclusion of tile tc (issued first so it can start early)
        if k > 1:
            if tc >= 0:
                _round_exclude(xf_c, vb_c, (tc % 2) == 0)
        # A: load + upcast + tag of tile s
        xf_new = sel
        if s < n_tiles:
            ra = _rows_of_tile(s, n_full, r_tile, r_rem)
            xf_new = _load_encode(x2, iot, p0, s * r_tile, ra, E, logits.dtype)
        # B: round 0 (max + exclusion mask) of tile tb
        vb_new = sel
        if tb >= 0:
            if tb < n_tiles:
                rb = _rows_of_tile(tb, n_full, r_tile, r_rem)
                _round_max(xf_b, sel, tb * r_tile, rb, E)
                if k > 1:
                    vb_new = _round_mask(xf_b, sel, tb * r_tile, rb, E)
        # C2: remaining rounds of tile tc
        if k > 1:
            if tc >= 0:
                rc = _rows_of_tile(tc, n_full, r_tile, r_rem)
                _rounds_after_first_exclusion(xf_c, sel, tc * r_tile, rc, E, Q, k,
                                              (tc % 2) == 0)
        xf_c = xf_b
        vb_c = vb_new
        xf_b = xf_new

    # decode: index = low tag bits, value = exact fp32 (tag bits cleared); exact int32 math
    si = sel.view(nl.int32)
    mk = nl.ndarray((PART, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=mk, value=idx_mask)
    itsm = nl.ndarray((PART, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=itsm, data1=si, data2=mk, op=nl.bitwise_and,
                       engine=nisa.engine.gpsimd)
    vbits = nl.ndarray((PART, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=vbits, data1=si, data2=itsm, op=nl.subtract,
                       engine=nisa.engine.gpsimd)
    vals = vbits.view(nl.float32)

    # indices in output layout (column q*k + j), int32 (4-byte aligned strided writes)
    io = nl.ndarray((PART, QK), dtype=nl.int32, buffer=nl.sbuf)
    for j in range(k):
        nisa.tensor_copy(dst=io[:, j:QK:k], src=itsm[:, j * Q:(j + 1) * Q])

    # softmax over the k selected values only (fp32)
    mx = nl.ndarray((PART, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mx, src=vals[:, 0:Q])
    for j in range(1, k):
        nisa.tensor_tensor(dst=mx, data1=mx, data2=vals[:, j * Q:(j + 1) * Q],
                           op=nl.maximum)
    dd = nl.ndarray((PART, QK), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(k):
        nisa.tensor_tensor(dst=dd[:, j * Q:(j + 1) * Q], data1=vals[:, j * Q:(j + 1) * Q],
                           data2=mx, op=nl.subtract)
    ex = nl.ndarray((PART, QK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ex, op=nl.exp, data=dd)
    ssum = nl.ndarray((PART, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ssum, src=ex[:, 0:Q])
    for j in range(1, k):
        nisa.tensor_tensor(dst=ssum, data1=ssum, data2=ex[:, j * Q:(j + 1) * Q], op=nl.add)
    # normalized weights interleaved in fp32 (aligned), then one contiguous cast
    w32 = nl.ndarray((PART, QK), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(k):
        nisa.tensor_tensor(dst=w32[:, j:QK:k], data1=ex[:, j * Q:(j + 1) * Q], data2=ssum,
                           op=nl.divide)
    wt = nl.ndarray((PART, QK), dtype=logits.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wt, src=w32)

    nisa.dma_copy(dst=w2[p0:p0 + PART, :], src=wt)
    nisa.dma_copy(dst=i2[p0:p0 + PART, :], src=io)
    return weights, indices


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    global _LAST_CONFIG
    m_rows = int(logits.shape[0])
    n_exp = int(logits.shape[1])
    kk = int(k)
    if logits.dtype == torch.float32:
        raise NotImplementedError("tag-bit argmax requires a 16-bit input dtype")
    if m_rows % (LNC * PART) != 0:
        raise ValueError("M must be a multiple of 256")
    if kk < 1 or kk > n_exp:
        raise ValueError("require 1 <= k <= E")
    idx_bits = max(1, (n_exp - 1).bit_length())
    if idx_bits > 13:
        raise ValueError("E too large for the tag-bit argmax")
    idx_mask = (1 << idx_bits) - 1
    q = m_rows // (LNC * PART)
    r_tile = 8
    if q <= 16:
        r_tile = 4
    if r_tile > q:
        r_tile = q
    n_full = q // r_tile
    r_rem = q % r_tile

    weights, indices = _moe_topk_gating_kernel(logits, kk, r_tile, idx_mask)

    _LAST_CONFIG = {
        "lnc": LNC,
        "rows_per_partition_per_program": q,
        "r_tile": r_tile,
        "n_full_tiles": n_full,
        "remainder_rows_per_partition": r_rem,
        "idx_bits": idx_bits,
        "pipeline": "3-stage(load/encode, round0, rest)",
        "exclude_add_engine": "alternate(gpsimd,vector)",
        "k": kk,
        "E": n_exp,
    }
    return weights, indices


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
