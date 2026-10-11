import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# destindex: out[dest_loc[t]] = kv[t] for the nope and the rope part.
#
# * Output buffers are allocated in run() with torch.empty_like and passed to
#   the kernel 16 times each (as distinct .view(-1) objects). Column c of the
#   scatter is written through alias c % 16, so the compiler only orders
#   scatters of the same alias (dynamic destinations -> conservative
#   write-after-write ordering) and up to 16 scatters per part are in flight.
#   Correct because dest_loc is a permutation: every scatter writes disjoint
#   rows and every output row is written exactly once.
# * Program pid (LNC=2) owns tokens [pid*T/2, (pid+1)*T/2); partition p /
#   column c <-> token pid*T/2 + p*NT + c, so all loads are contiguous.
# * Nope and rope rows of a batch are streamed HBM -> SBUF into one slot of a
#   6-slot ring buffer (nope part split into 4 partition-group DMAs), with a
#   ramped batch schedule (1,1,2,4 tokens per partition, then 8); loads are
#   issued 4 batches ahead.
# * Each 128-token column is scattered SBUF -> HBM with one indirect DMA per
#   part (vector_offset = destination row per partition).
# o_* is never read; no task input is written.
# ---------------------------------------------------------------------------

_P = 128
_LNC = 2          # a bare kernel call runs with LNC=2 on this host
_KMAX = 8         # max tokens per partition per batch (schedule assumes 8)
_NBUF = 6         # ring-buffer depth (max 6, see _pick6)
_C0 = 8           # columns in the first (early) index piece
_NALIAS = 16      # output aliases per part (fixed: 16 kernel args each)
_LSPLIT = 4       # partition groups per nope batch load

_LAST_CONFIG = {}


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


def _pick16(s, a0, a1, a2, a3, a4, a5, a6, a7,
            a8, a9, a10, a11, a12, a13, a14, a15):
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
    if s == 5:
        return a5
    if s == 6:
        return a6
    if s == 7:
        return a7
    if s == 8:
        return a8
    if s == 9:
        return a9
    if s == 10:
        return a10
    if s == 11:
        return a11
    if s == 12:
        return a12
    if s == 13:
        return a13
    if s == 14:
        return a14
    return a15


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


def _load_batch(bufl, src_n, src_r, pid, ls, lz, Fn, Fr, RO, PG, nsplit):
    # nope rows: split over partition groups so several DMAs run in parallel
    for g in range(nsplit):
        p0 = g * PG
        nisa.dma_copy(dst=bufl[p0:p0 + PG, 0:lz * Fn],
                      src=src_n[pid, p0:p0 + PG, ls * Fn:(ls + lz) * Fn])
    # rope rows of the same tokens, into the rope area of the same slot
    nisa.dma_copy(dst=bufl[:, RO:RO + lz * Fr],
                  src=src_r[pid, :, ls * Fr:(ls + lz) * Fr])


@nki.jit
def _destindex_kernel(kv_nope, kv_rope, dest_loc,
                      on0, on1, on2, on3, on4, on5, on6, on7,
                      on8, on9, on10, on11, on12, on13, on14, on15,
                      or0, or1, or2, or3, or4, or5, or6, or7,
                      or8, or9, or10, or11, or12, or13, or14, or15,
                      nbuf, c0, nsplit):
    P = 128
    KMAX = 8
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]
    RO = KMAX * Fn                 # rope offset inside a ring slot
    W = KMAX * (Fn + Fr)           # ring slot width (elements per partition)
    PG = P // nsplit

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)          # tokens per partition for this program
    NB = NT // KMAX + 3            # number of batches (ramped schedule)
    D = min(NB, nbuf - 2)          # prefetch distance

    done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    # source views
    src_n = kv_nope.reshape((nprog, P, NT * Fn))
    src_r = kv_rope.reshape((nprog, P, NT * Fr))
    idx_v = dest_loc.reshape((nprog, P, NT))

    # first index piece (small, so the first scatters can start early)
    ca = min(c0, NT)
    idx_a = nl.ndarray((P, ca), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_a, src=idx_v[pid, :, 0:ca])

    # ring buffer: each slot holds [KMAX nope rows | KMAX rope rows] per partition
    buf0 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf1 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf2 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf3 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf4 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf5 = nl.ndarray((P, W), dtype=kv_nope.dtype, buffer=nl.sbuf)

    # first batch
    _load_batch(buf0, src_n, src_r, pid, _bstart(0), _bsize(0),
                Fn, Fr, RO, PG, nsplit)

    # remaining index columns
    idx_b = idx_a
    if NT > ca:
        idx_b = nl.ndarray((P, NT - ca), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_b, src=idx_v[pid, :, ca:NT])

    # prologue: prefetch batches 1 .. D-1
    for bl in range(1, D):
        bufl = _pick6(bl % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
        _load_batch(bufl, src_n, src_r, pid, _bstart(bl), _bsize(bl),
                    Fn, Fr, RO, PG, nsplit)

    for b in range(NB):
        # prefetch batch b + D (its slot was last used by batch b + D - nbuf <= b - 2)
        if b + D < NB:
            bl = b + D
            bufl = _pick6(bl % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
            _load_batch(bufl, src_n, src_r, pid, _bstart(bl), _bsize(bl),
                        Fn, Fr, RO, PG, nsplit)

        buf = _pick6(b % nbuf, buf0, buf1, buf2, buf3, buf4, buf5)
        bs = _bstart(b)
        bz = _bsize(b)
        for j in range(bz):
            c = bs + j
            iv = _col(idx_a, idx_b, c, ca, 1)
            a = c % 16
            dn = _pick16(a, on0, on1, on2, on3, on4, on5, on6, on7,
                         on8, on9, on10, on11, on12, on13, on14, on15).reshape((T, Fn))
            dr = _pick16(a, or0, or1, or2, or3, or4, or5, or6, or7,
                         or8, or9, or10, or11, or12, or13, or14, or15).reshape((T, Fr))
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
                src=buf[:, RO + j * Fr:RO + (j + 1) * Fr],
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

    nviews = [out_nope.view(-1) for _ in range(_NALIAS)]
    rviews = [out_rope.view(-1) for _ in range(_NALIAS)]

    _destindex_kernel(kv_nope, kv_rope, dest_loc, *nviews, *rviews,
                      _NBUF, _C0, _LSPLIT)

    _LAST_CONFIG = {
        "lnc": _LNC,
        "partitions": _P,
        "tokens_per_partition": nt,
        "batch_schedule": "1,1,2,4 then 8 tokens/partition",
        "ring_buffers": _NBUF,
        "prefetch_distance": min(nt // _KMAX + 3, _NBUF - 2),
        "nope_load_partition_groups": _LSPLIT,
        "rope_staging": "per batch, in the nope ring slot",
        "early_index_cols": min(_C0, nt),
        "output_aliases_per_part": _NALIAS,
        "scatter": "indirect_dma_vector_offset_per_128_tokens_round_robin_aliases",
    }
    return out_nope, out_rope


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
