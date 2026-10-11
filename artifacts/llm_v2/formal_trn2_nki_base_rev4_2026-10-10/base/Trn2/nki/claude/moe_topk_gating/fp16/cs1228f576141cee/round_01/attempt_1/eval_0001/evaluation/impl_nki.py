"""
moe_topk_gating for NKI 0.7.0b1 on Trn2 (LNC=2, one logical core = 2 physical cores).

Layout: each program (physical core) owns M/2 rows. Partition p of program `pid` owns
Q = M/256 consecutive rows. They are processed in tiles of R rows per partition, so one
tile is a [128, R*E] SBUF block.

Per tile (fused, everything stays on chip):
  * Row load: one DMA of fp16 logits, then an exact upcast to fp32 (ScalarE).
  * Argmax tagging: the fp16->fp32 upcast leaves the low 13 mantissa bits at zero, so the
    column index (7 bits for E=128) is OR-ed into those bits (GpSimd, exact int32).
    Distinct values keep their float32 order, so one fp32 max-reduction returns both the
    exact maximum (high bits) and its column (low bits). This is a fused max+argmax.
  * k rounds. Each round does a full-row fp32 max over the remaining row (VectorE
    tensor_reduce over the E axis) and records it in slot j. If another round follows,
    the selected column is overwritten with -inf:
      v = (row >= max) * (-FLT_MAX), stored as bf16 (RNE turns -FLT_MAX into -inf, and
      0 stays -0), then row = row + v. The selected column becomes -inf and every other
      column is unchanged bit for bit.
  * The k selected tagged values of every row stay in an SBUF tile.
After all tiles: decode index = bits & mask and value = bits - index, which is the exact
fp32 value. The softmax over the k values is done in fp32: subtract the max, exp, sum,
divide. Weights are cast to fp16 on write. Both outputs are stored with one DMA each.
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
R_TILE = 8                                  # rows per partition per main tile
NEG_LOWEST_F32 = -3.4028234663852886e38     # -FLT_MAX; rounds to -inf when stored as bf16

_LAST_CONFIG = {}


def _load_encode(x2, iot, p0, q0, R, E, in_dtype):
    # Row load (one traversal), exact upcast to fp32, tag low mantissa bits with column.
    F = R * E
    xh = nl.ndarray((128, F), dtype=in_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xh, src=x2[p0:p0 + 128, q0 * E:q0 * E + F])
    xf = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xf, src=xh, engine=nisa.engine.scalar)
    xi = xf.view(nl.int32)
    nisa.tensor_tensor(dst=xi, data1=xi, data2=iot[:, 0:F], op=nl.bitwise_or)
    return xf


def _round_max(xf, sel, q0, R, E, k, j):
    # Full-row fp32 max (max + argmax via tag bits) for R rows -> slot j of those rows.
    nisa.tensor_reduce(dst=sel[:, q0 * k + j:(q0 + R) * k:k], op=nl.maximum,
                       data=xf.reshape((128, R, E)), axis=2)


def _round_mask(xf, sel, q0, R, E, k, j):
    # v = (row >= selected max) * (-FLT_MAX) -> bf16: -inf at the selected column, -0 elsewhere.
    vb = nl.ndarray((128, R * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    for r in range(R):
        c = (q0 + r) * k + j
        nisa.tensor_scalar(dst=vb[:, r * E:(r + 1) * E], data=xf[:, r * E:(r + 1) * E],
                           op0=nl.greater_equal, operand0=sel[:, c:c + 1],
                           op1=nl.multiply, operand1=NEG_LOWEST_F32)
    return vb


def _round_exclude(xf, vb):
    # Overwrite the selected column with -inf (others: x + (-0) == x exactly).
    nisa.tensor_tensor(dst=xf, data1=xf, data2=vb, op=nl.add, engine=nisa.engine.gpsimd)


def _finish_tile(xf, vb, sel, q0, R, E, k):
    cur = vb
    for j in range(1, k):
        _round_exclude(xf, cur)
        _round_max(xf, sel, q0, R, E, k, j)
        if j < k - 1:
            cur = _round_mask(xf, sel, q0, R, E, k, j)


@nki.jit
def _moe_topk_gating_kernel(logits, k, r_tile, idx_mask):
    M, E = logits.shape
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    P = 128
    Q = M // (n_prog * P)          # rows per partition handled by this program
    QK = Q * k

    weights = nl.ndarray((M, k), dtype=logits.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray((M, k), dtype=nl.int32, buffer=nl.shared_hbm)

    x2 = logits.reshape((n_prog * P, Q * E))
    w2 = weights.reshape((n_prog * P, QK))
    i2 = indices.reshape((n_prog * P, QK))
    p0 = pid * P

    n_full = Q // r_tile
    r_rem = Q - n_full * r_tile
    n_tiles = n_full
    if r_rem > 0:
        n_tiles = n_full + 1
    r_alloc = r_tile
    if n_full == 0:
        r_alloc = r_rem

    # column index pattern 0..E-1 repeated for each row slot of a tile
    iot = nl.ndarray((P, r_alloc * E), dtype=nl.int32, buffer=nl.sbuf)
    for r in range(r_alloc):
        nisa.iota(dst=iot[:, r * E:(r + 1) * E], pattern=[[1, E]], offset=0,
                  channel_multiplier=0)

    # selected (tagged) values: column q*k + j = slot j of local row q
    sel = nl.ndarray((P, QK), dtype=nl.float32, buffer=nl.sbuf)

    # software-pipelined emission: round 0 of tile t, then remaining rounds of tile t-1
    prev_xf = sel
    prev_vb = sel
    prev_q0 = 0
    prev_R = 0
    for t in range(n_tiles):
        R = r_tile
        if t >= n_full:
            R = r_rem
        q0 = t * r_tile
        xf = _load_encode(x2, iot, p0, q0, R, E, logits.dtype)
        _round_max(xf, sel, q0, R, E, k, 0)
        vb = xf
        if k > 1:
            vb = _round_mask(xf, sel, q0, R, E, k, 0)
        if t > 0:
            _finish_tile(prev_xf, prev_vb, sel, prev_q0, prev_R, E, k)
        prev_xf = xf
        prev_vb = vb
        prev_q0 = q0
        prev_R = R
    _finish_tile(prev_xf, prev_vb, sel, prev_q0, prev_R, E, k)

    # decode: index = low tag bits, value = exact fp32 (tag bits cleared)
    si = sel.view(nl.int32)
    mk = nl.ndarray((P, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=mk, value=idx_mask)
    it = nl.ndarray((P, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=it, data1=si, data2=mk, op=nl.bitwise_and)
    vbits = nl.ndarray((P, QK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=vbits, data1=si, data2=it, op=nl.subtract)
    vals = vbits.view(nl.float32)

    # softmax over the k selected values (fp32)
    mx = nl.ndarray((P, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mx, src=vals[:, 0:QK:k])
    for j in range(1, k):
        nisa.tensor_tensor(dst=mx, data1=mx, data2=vals[:, j:QK:k], op=nl.maximum)
    dd = nl.ndarray((P, QK), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(k):
        nisa.tensor_tensor(dst=dd[:, j:QK:k], data1=vals[:, j:QK:k], data2=mx,
                           op=nl.subtract)
    ex = nl.ndarray((P, QK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ex, op=nl.exp, data=dd)
    ssum = nl.ndarray((P, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ssum, src=ex[:, 0:QK:k])
    for j in range(1, k):
        nisa.tensor_tensor(dst=ssum, data1=ssum, data2=ex[:, j:QK:k], op=nl.add)
    wt = nl.ndarray((P, QK), dtype=logits.dtype, buffer=nl.sbuf)
    for j in range(k):
        nisa.tensor_tensor(dst=wt[:, j:QK:k], data1=ex[:, j:QK:k], data2=ssum,
                           op=nl.divide)

    nisa.dma_copy(dst=w2[p0:p0 + P, :], src=wt)
    nisa.dma_copy(dst=i2[p0:p0 + P, :], src=it)
    return weights, indices


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    global _LAST_CONFIG
    m_rows = int(logits.shape[0])
    n_exp = int(logits.shape[1])
    kk = int(k)
    lnc = 2
    if logits.dtype == torch.float32:
        raise NotImplementedError("tag-bit argmax requires a 16-bit input dtype")
    if m_rows % (lnc * PART) != 0:
        raise ValueError("M must be a multiple of 256")
    if kk < 1 or kk > n_exp:
        raise ValueError("require 1 <= k <= E")
    idx_bits = max(1, (n_exp - 1).bit_length())
    if idx_bits > 13:
        raise ValueError("E too large for the tag-bit argmax")
    idx_mask = (1 << idx_bits) - 1
    q = m_rows // (lnc * PART)
    r_tile = R_TILE
    n_full = q // r_tile
    r_rem = q % r_tile

    weights, indices = _moe_topk_gating_kernel(logits, kk, r_tile, idx_mask)

    _LAST_CONFIG = {
        "lnc": lnc,
        "rows_per_partition_per_program": q,
        "r_tile": r_tile,
        "n_full_tiles": n_full,
        "remainder_rows_per_partition": r_rem,
        "idx_bits": idx_bits,
        "k": kk,
        "E": n_exp,
    }
    return weights, indices


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
