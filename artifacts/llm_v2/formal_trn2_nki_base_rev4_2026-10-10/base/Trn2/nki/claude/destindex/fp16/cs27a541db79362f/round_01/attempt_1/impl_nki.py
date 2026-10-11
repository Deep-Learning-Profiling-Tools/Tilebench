import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# destindex: out[dest_loc[t]] = kv[t] for the nope and the rope part.
#
# Mapping (per logical core, LNC=2):
#   * program `pid` owns tokens [pid*T/2, (pid+1)*T/2)
#   * inside a program, partition p / column c  <->  token pid*T/2 + p*NT + c
#     (NT = T / (2*128)), so every HBM load is contiguous per partition.
#   * all indices of the program are loaded once into an SBUF [128, NT] int32 tile
#   * all rope rows of the program are loaded once into SBUF [128, NT*Fr]
#   * nope rows are streamed in batches of kb tokens per partition through a
#     4-slot SBUF ring buffer
#   * each 128-token column c is scattered to HBM with an indirect DMA whose
#     destination row for partition p is idx[p, c] (vector_offset).
# The output buffers are freshly allocated by the kernel and fully overwritten
# (dest_loc is a permutation), o_* is never read and nothing is written to inputs.
# ---------------------------------------------------------------------------

_P = 128
_NBUF = 4          # ring-buffer depth for the nope stream (max 4, see _pick)
_MAX_KB = 8        # max tokens per partition per nope load
_LNC = 2           # bare kernel call runs with LNC=2 on this host

_LAST_CONFIG = {}


def _pick(s, b0, b1, b2, b3):
    if s == 0:
        return b0
    if s == 1:
        return b1
    if s == 2:
        return b2
    return b3


@nki.jit
def _destindex_kernel(kv_nope, kv_rope, dest_loc, kb, nbuf):
    P = 128
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]

    out_nope = nl.ndarray(kv_nope.shape, dtype=kv_nope.dtype, buffer=nl.shared_hbm)
    out_rope = nl.ndarray(kv_rope.shape, dtype=kv_rope.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)          # tokens per partition handled by this program
    NB = NT // kb                  # number of nope batches

    # HBM views
    on2 = out_nope.reshape((T, Fn))
    or2 = out_rope.reshape((T, Fr))
    src_n = kv_nope.reshape((nprog, P, NT * Fn))
    src_r = kv_rope.reshape((nprog, P, NT * Fr))
    idx_v = dest_loc.reshape((nprog, P, NT))

    # all destination indices of this program: idx[p, c] = dest_loc[base + p*NT + c]
    idx = nl.ndarray((P, NT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx, src=idx_v[pid])

    # all rope rows of this program
    rope_t = nl.ndarray((P, NT * Fr), dtype=kv_rope.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rope_t, src=src_r[pid])

    # ring buffer for nope batches
    buf0 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf1 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf2 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf3 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)

    for b in range(NB):
        buf = _pick(b % nbuf, buf0, buf1, buf2, buf3)
        nisa.dma_copy(dst=buf, src=src_n[pid, :, b * kb * Fn:(b + 1) * kb * Fn])
        for j in range(kb):
            c = b * kb + j
            # scatter nope rows: partition p -> row idx[p, c]
            nisa.dma_copy(
                dst=on2.ap(pattern=[[0, P], [1, Fn]], offset=0,
                           vector_offset=idx[:, c:c + 1], indirect_dim=0),
                src=buf[:, j * Fn:(j + 1) * Fn],
            )
            # scatter rope rows: partition p -> row idx[p, c]
            nisa.dma_copy(
                dst=or2.ap(pattern=[[0, P], [1, Fr]], offset=0,
                           vector_offset=idx[:, c:c + 1], indirect_dim=0),
                src=rope_t[:, c * Fr:(c + 1) * Fr],
            )

    return out_nope, out_rope


def _choose_kb(nt):
    kb = 1
    c = 1
    while c <= _MAX_KB:
        if nt % c == 0:
            kb = c
        c += 1
    return kb


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    global _LAST_CONFIG
    T = kv_nope.shape[0]
    assert T % (_LNC * _P) == 0, "T must be a multiple of 256 (task domain: multiple of 2048)"
    nt = T // (_LNC * _P)
    kb = _choose_kb(nt)
    nbuf = _NBUF

    out_nope, out_rope = _destindex_kernel(kv_nope, kv_rope, dest_loc, kb, nbuf)

    _LAST_CONFIG = {
        "lnc": _LNC,
        "partitions": _P,
        "tokens_per_partition": nt,
        "nope_tokens_per_partition_per_load": kb,
        "nope_ring_buffers": nbuf,
        "scatter": "indirect_dma_vector_offset_per_128_tokens",
    }
    return out_nope, out_rope


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
