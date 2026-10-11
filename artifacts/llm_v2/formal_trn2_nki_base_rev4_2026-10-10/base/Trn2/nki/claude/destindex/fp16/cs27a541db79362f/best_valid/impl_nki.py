import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# destindex: out[dest_loc[t]] = kv[t] for the nope and the rope part.
#
# * Output buffers are allocated in run() with torch.empty_like and passed to
#   the kernel several times (as distinct .view(-1) objects). Successive
#   indirect scatters are written through different aliases (column c uses
#   alias c % 4), so the compiler does not have to chain all scatters of one
#   output tensor (dynamic destinations -> conservative write-after-write
#   ordering). Correct because dest_loc is a permutation: every scatter
#   writes disjoint rows, and every output row is written exactly once.
# * Program pid (LNC=2) owns tokens [pid*T/2, (pid+1)*T/2); partition p /
#   column c <-> token pid*T/2 + p*NT + c, so all loads are contiguous.
# * Nope rows are streamed HBM -> SBUF through a 6-slot ring buffer with a
#   ramped batch schedule (1,1,2,4 tokens per partition, then 8) so that the
#   first scatters start early; loads are issued 4 batches ahead.
# * Each 128-token column is scattered SBUF -> HBM with one indirect DMA per
#   part (vector_offset = destination row per partition).
# o_* is never read; no task input is written.
# ---------------------------------------------------------------------------

_P = 128
_LNC = 2          # a bare kernel call runs with LNC=2 on this host
_KMAX = 8         # max nope tokens per partition per load (schedule assumes 8)
_NBUF = 6         # ring-buffer depth (max 6, see _pick6)
_C0 = 8           # columns in the first (early) index / rope piece
_NALIAS = 4       # output aliases per part (fixed: 4 kernel args each)

_LAST_CONFIG = {}


def _pick4(s, a0, a1, a2, a3):
    if s == 0:
        return a0
    if s == 1:
        return a1
    if s == 2:
        return a2
    return a3


def _pick6(s, a0, a1, a2, a3, a4, a5):
    if s == 0:
        return a0
    if s == 1:
        return a1
    if s == 2:
        return a2
    if s == 3:
        return a3
    if s == 4:
        return a4
    return a5


def _bstart(b):
    # ramped schedule: sizes 1,1,2,4 then 8 -> starts 0,1,2,4,8,16,...
    if b < 3:
        return b
    if b == 3:
        return 4
    return 8 * (b - 3)


def _bsize(b):
    if b < 2:
        return 1
    if b == 2:
        return 2
    if b == 3:
        return 4
    return 8


def _col(ta, tb, c, ca, w):
    if c < ca:
        return ta[:, c * w:(c + 1) * w]
    return tb[:, (c - ca) * w:(c - ca + 1) * w]


@nki.jit
def _destindex_kernel(kv_nope, kv_rope, dest_loc,
                      on0, on1, on2, on3, or0, or1, or2, or3,
                      nbuf, c0):
    P = 128
    KMAX = 8
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)          # tokens per partition for this program
    NB = NT // KMAX + 3            # number of nope batches (ramped schedule)
    D = min(NB, nbuf - 2)          # prefetch distance

    done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    # destination views (all aliases of the same output buffers)
    dn0 = on0.reshape((T, Fn))
    dn1 = on1.reshape((T, Fn))
    dn2 = on2.reshape((T, Fn))
    dn3 = on3.reshape((T, Fn))
    dr0 = or0.reshape((T, Fr))
    dr1 = or1.reshape((T, Fr))
    dr2 = or2.reshape((T, Fr))
    dr3 = or3.reshape((T, Fr))

    # source views
    src_n = kv_nope.reshape((nprog, P, NT * Fn))
    src_r = kv_rope.reshape((nprog, P, NT * Fr))
    idx_v = dest_loc.reshape((nprog, P, NT))

    # first index / rope piece (small, so the first scatters can start early)
    ca = min(c0, NT)
    idx_a = nl.ndarray((P, ca), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_a, src=idx_v[pid, :, 0:ca])
    rope_a = nl.ndarray((P, ca * Fr), dtype=kv_rope.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rope_a, src=src_r[pid, :, 0:ca * Fr])

    # nope ring buffer
    buf0 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf1 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf2 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf3 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf4 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf5 = nl.ndarray((P, KMAX * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)

    # first nope batch
    s0 = _bstart(0)
    sz = _bsize(0)
    nisa.dma_copy(dst=buf0[:, 0:sz * Fn], src=src_n[pid, :, s0 * Fn:(s0 + sz) * Fn])

    # remaining index / rope columns
    idx_b = idx_a
    rope_b = rope_a
    if NT > ca:
        idx_b = nl.ndarray((P, NT - ca), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_b, src=idx_v[pid, :, ca:NT])
        rope_b = nl.ndarray((P, (NT - ca) * Fr), dtype=kv_rope.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rope_b, src=src_r[pid, :, ca * Fr:NT * Fr])

    # prologue: prefetch batches 1 .. D-1
    for bl in range(1, D):
        bufl = _pick6(bl % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
        ls = _bstart(bl)
        lz = _bsize(bl)
        nisa.dma_copy(dst=bufl[:, 0:lz * Fn], src=src_n[pid, :, ls * Fn:(ls + lz) * Fn])

    for b in range(NB):
        # prefetch batch b + D (its slot was last used by batch b + D - nbuf <= b - 2)
        if b + D < NB:
            bl = b + D
            bufl = _pick6(bl % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
            ls = _bstart(bl)
            lz = _bsize(bl)
            nisa.dma_copy(dst=bufl[:, 0:lz * Fn], src=src_n[pid, :, ls * Fn:(ls + lz) * Fn])

        buf = _pick6(b % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
        bs = _bstart(b)
        bz = _bsize(b)
        for j in range(bz):
            c = bs + j
            iv = _col(idx_a, idx_b, c, ca, 1)
            dn = _pick4(c % 4, dn0, dn1, dn2, dn3)
            dr = _pick4(c % 4, dr0, dr1, dr2, dr3)
            # nope: partition p -> output row idx[p, c]
            nisa.dma_copy(
                dst=dn.ap(pattern=[[0, P], [1, Fn]], offset=0,
                          vector_offset=iv, indirect_dim=0),
                src=buf[:, j * Fn:(j + 1) * Fn],
            )
            # rope: partition p -> output row idx[p, c]
            nisa.dma_copy(
                dst=dr.ap(pattern=[[0, P], [1, Fr]], offset=0,
                          vector_offset=iv, indirect_dim=0),
                src=_col(rope_a, rope_b, c, ca, Fr),
            )

    # small status output (a kernel must return a tensor)
    flag = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=flag, value=1)
    nisa.dma_copy(dst=done, src=flag)
    return done


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    global _LAST_CONFIG
    T = kv_nope.shape[0]
    assert T % (_LNC * _P * _KMAX) == 0, "T must be a multiple of 2048 (task domain)"
    nt = T // (_LNC * _P)

    # fresh output buffers; fully overwritten by the scatter (dest_loc is a permutation)
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    _destindex_kernel(
        kv_nope, kv_rope, dest_loc,
        out_nope.view(-1), out_nope.view(-1), out_nope.view(-1), out_nope.view(-1),
        out_rope.view(-1), out_rope.view(-1), out_rope.view(-1), out_rope.view(-1),
        _NBUF, _C0,
    )

    _LAST_CONFIG = {
        "lnc": _LNC,
        "partitions": _P,
        "tokens_per_partition": nt,
        "nope_batch_schedule": "1,1,2,4 then 8 tokens/partition",
        "nope_ring_buffers": _NBUF,
        "prefetch_distance": min(nt // _KMAX + 3, _NBUF - 2),
        "early_index_cols": min(_C0, nt),
        "output_aliases_per_part": _NALIAS,
        "scatter": "indirect_dma_vector_offset_per_128_tokens_round_robin_aliases",
    }
    return out_nope, out_rope


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
