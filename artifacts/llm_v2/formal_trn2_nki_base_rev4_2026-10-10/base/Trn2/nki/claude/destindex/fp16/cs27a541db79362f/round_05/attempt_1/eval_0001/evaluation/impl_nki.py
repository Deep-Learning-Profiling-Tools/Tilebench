import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# destindex: out[dest_loc[t]] = kv[t] for the nope and the rope part.
#
# * Output buffers are allocated in run() with torch.empty_like and passed to
#   the kernel several times (as distinct .view(-1) objects). Successive
#   indirect scatters are written through different aliases (nope column c
#   uses alias c % 8, rope column c uses alias c % 4), so the compiler does
#   not have to chain all scatters of one output tensor. Correct because
#   dest_loc is a permutation: every scatter writes disjoint rows and every
#   output row is written exactly once.
# * Program pid (LNC=2) owns tokens [pid*T/2, (pid+1)*T/2); partition p /
#   column c <-> token pid*T/2 + p*NT + c, so index / rope loads are
#   contiguous per partition.
# * Nope part (96% of the bytes): each 128-token column is scattered
#   directly HBM -> HBM with one indirect DMA (vector_offset = destination
#   row per partition). No SBUF staging: each byte passes the DMA engines
#   once.
# * Rope part (128-byte rows): staged in SBUF with two contiguous loads,
#   then scattered SBUF -> HBM per column with one indirect DMA.
# o_* is never read; no task input is written.
# ---------------------------------------------------------------------------

_P = 128
_LNC = 2          # a bare kernel call runs with LNC=2 on this host
_C0 = 8           # columns in the first (early) index / rope piece
_NALIAS_N = 8     # nope output aliases (fixed: 8 kernel args)
_NALIAS_R = 4     # rope output aliases (fixed: 4 kernel args)

_LAST_CONFIG = {}


def _pick4(s, a0, a1, a2, a3):
    if s == 0:
        return a0
    if s == 1:
        return a1
    if s == 2:
        return a2
    return a3


def _pick8(s, a0, a1, a2, a3, a4, a5, a6, a7):
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
    return a7


def _col(ta, tb, c, ca, w):
    if c < ca:
        return ta[:, c * w:(c + 1) * w]
    return tb[:, (c - ca) * w:(c - ca + 1) * w]


@nki.jit
def _destindex_kernel(kv_nope, kv_rope, dest_loc,
                      on0, on1, on2, on3, on4, on5, on6, on7,
                      or0, or1, or2, or3,
                      c0):
    P = 128
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)          # tokens per partition for this program

    done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    # destination views (all aliases of the same output buffers)
    dn0 = on0.reshape((T, Fn))
    dn1 = on1.reshape((T, Fn))
    dn2 = on2.reshape((T, Fn))
    dn3 = on3.reshape((T, Fn))
    dn4 = on4.reshape((T, Fn))
    dn5 = on5.reshape((T, Fn))
    dn6 = on6.reshape((T, Fn))
    dn7 = on7.reshape((T, Fn))
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

    # remaining index / rope columns
    idx_b = idx_a
    rope_b = rope_a
    if NT > ca:
        idx_b = nl.ndarray((P, NT - ca), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_b, src=idx_v[pid, :, ca:NT])
        rope_b = nl.ndarray((P, (NT - ca) * Fr), dtype=kv_rope.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rope_b, src=src_r[pid, :, ca * Fr:NT * Fr])

    for c in range(NT):
        iv = _col(idx_a, idx_b, c, ca, 1)
        dn = _pick8(c % 8, dn0, dn1, dn2, dn3, dn4, dn5, dn6, dn7)
        dr = _pick4(c % 4, dr0, dr1, dr2, dr3)
        # nope: HBM rows of 128 tokens -> output row idx[p, c] (direct HBM -> HBM)
        nisa.dma_copy(
            dst=dn.ap(pattern=[[0, P], [1, Fn]], offset=0,
                      vector_offset=iv, indirect_dim=0),
            src=src_n[pid, :, c * Fn:(c + 1) * Fn],
        )
        # rope: staged SBUF column -> output row idx[p, c]
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
    assert T % (_LNC * _P) == 0, "T must be a multiple of 256 (task domain: 2048)"
    nt = T // (_LNC * _P)

    # fresh output buffers; fully overwritten by the scatter (dest_loc is a permutation)
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    _destindex_kernel(
        kv_nope, kv_rope, dest_loc,
        out_nope.view(-1), out_nope.view(-1), out_nope.view(-1), out_nope.view(-1),
        out_nope.view(-1), out_nope.view(-1), out_nope.view(-1), out_nope.view(-1),
        out_rope.view(-1), out_rope.view(-1), out_rope.view(-1), out_rope.view(-1),
        _C0,
    )

    _LAST_CONFIG = {
        "lnc": _LNC,
        "partitions": _P,
        "tokens_per_partition": nt,
        "nope_scatter": "direct_hbm_to_hbm_indirect_dma_per_128_tokens",
        "rope_scatter": "sbuf_staged_indirect_dma_per_128_tokens",
        "early_index_cols": min(_C0, nt),
        "nope_output_aliases": _NALIAS_N,
        "rope_output_aliases": _NALIAS_R,
    }
    return out_nope, out_rope


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
