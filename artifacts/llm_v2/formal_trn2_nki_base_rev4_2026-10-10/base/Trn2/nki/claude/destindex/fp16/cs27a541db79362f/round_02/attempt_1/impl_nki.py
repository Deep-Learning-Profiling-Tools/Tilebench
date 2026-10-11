import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# destindex: out[dest_loc[t]] = kv[t] for the nope and the rope part.
#
# Primary path ("hbm_to_hbm_indirect"):
#   * program `pid` (LNC=2) owns tokens [pid*T/2, (pid+1)*T/2)
#   * row p / column c  <->  token pid*T/2 + p*NT + c   (NT = T / 256)
#   * the destination indices of the program are loaded once into SBUF
#     (two tiles so the first scatters can start early)
#   * for each column c one indirect DMA copies the 128 source rows directly
#     HBM -> HBM to rows idx[p, c] of the output (vector_offset scatter),
#     once for nope and once for rope. No SBUF staging of the payload.
# Fallback path ("staged", only if the primary path is not supported by the
# compiler for this shape; decided once per shape, deterministic):
#   rows are staged through an SBUF ring buffer and scattered from SBUF.
# Outputs are freshly allocated in the kernel and fully overwritten
# (dest_loc is a permutation); o_* is never read and inputs are never written.
# ---------------------------------------------------------------------------

_P = 128
_LNC = 2           # a bare kernel call runs with LNC=2 on this host
_NBUF = 4          # staged path: ring-buffer depth (max 4, see _pick)
_MAX_KB = 8        # staged path: max tokens per partition per load
_IDX_HEAD = 8      # primary path: index columns in the first (early) index tile

_LAST_CONFIG = {}
_PATH_CACHE = {}


def _pick(s, b0, b1, b2, b3):
    if s == 0:
        return b0
    if s == 1:
        return b1
    if s == 2:
        return b2
    return b3


def _idx_col(idx_a, idx_b, c, c0):
    if c < c0:
        return idx_a[:, c:c + 1]
    return idx_b[:, c - c0:c - c0 + 1]


@nki.jit
def _destindex_h2h_kernel(kv_nope, kv_rope, dest_loc, c0):
    P = 128
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]

    out_nope = nl.ndarray(kv_nope.shape, dtype=kv_nope.dtype, buffer=nl.shared_hbm)
    out_rope = nl.ndarray(kv_rope.shape, dtype=kv_rope.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)          # tokens per row handled by this program

    # HBM views
    on2 = out_nope.reshape((T, Fn))
    or2 = out_rope.reshape((T, Fr))
    src_n = kv_nope.reshape((nprog, P, NT, Fn))
    src_r = kv_rope.reshape((nprog, P, NT, Fr))
    idx_v = dest_loc.reshape((nprog, P, NT))

    # destination indices: idx[p, c] = dest_loc[base + p*NT + c]
    idx_a = nl.ndarray((P, c0), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_a, src=idx_v[pid, :, 0:c0])
    idx_b = nl.ndarray((P, NT - c0), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_b, src=idx_v[pid, :, c0:NT])

    for c in range(NT):
        iv = _idx_col(idx_a, idx_b, c, c0)
        # nope: source row (base + p*NT + c) -> output row idx[p, c]
        nisa.dma_copy(
            dst=on2.ap(pattern=[[0, P], [1, Fn]], offset=0,
                       vector_offset=iv, indirect_dim=0),
            src=src_n[pid, :, c, :],
        )
        # rope: same mapping
        nisa.dma_copy(
            dst=or2.ap(pattern=[[0, P], [1, Fr]], offset=0,
                       vector_offset=iv, indirect_dim=0),
            src=src_r[pid, :, c, :],
        )

    return out_nope, out_rope


@nki.jit
def _destindex_staged_kernel(kv_nope, kv_rope, dest_loc, kb, nbuf):
    P = 128
    T = kv_nope.shape[0]
    Fn = kv_nope.shape[1] * kv_nope.shape[2]
    Fr = kv_rope.shape[1] * kv_rope.shape[2]

    out_nope = nl.ndarray(kv_nope.shape, dtype=kv_nope.dtype, buffer=nl.shared_hbm)
    out_rope = nl.ndarray(kv_rope.shape, dtype=kv_rope.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = T // (nprog * P)
    NB = NT // kb

    on2 = out_nope.reshape((T, Fn))
    or2 = out_rope.reshape((T, Fr))
    src_n = kv_nope.reshape((nprog, P, NT * Fn))
    src_r = kv_rope.reshape((nprog, P, NT * Fr))
    idx_v = dest_loc.reshape((nprog, P, NT))

    idx = nl.ndarray((P, NT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx, src=idx_v[pid])

    rope_t = nl.ndarray((P, NT * Fr), dtype=kv_rope.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rope_t, src=src_r[pid])

    buf0 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf1 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf2 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)
    buf3 = nl.ndarray((P, kb * Fn), dtype=kv_nope.dtype, buffer=nl.sbuf)

    for b in range(NB):
        buf = _pick(b % nbuf, buf0, buf1, buf2, buf3)
        nisa.dma_copy(dst=buf, src=src_n[pid, :, b * kb * Fn:(b + 1) * kb * Fn])
        for j in range(kb):
            c = b * kb + j
            nisa.dma_copy(
                dst=on2.ap(pattern=[[0, P], [1, Fn]], offset=0,
                           vector_offset=idx[:, c:c + 1], indirect_dim=0),
                src=buf[:, j * Fn:(j + 1) * Fn],
            )
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

    key = (tuple(kv_nope.shape), tuple(kv_rope.shape), str(kv_nope.dtype))
    path = _PATH_CACHE.get(key, "hbm_to_hbm_indirect" if nt >= 2 else "staged")

    result = None
    c0 = min(_IDX_HEAD, nt // 2)
    if path == "hbm_to_hbm_indirect":
        try:
            result = _destindex_h2h_kernel(kv_nope, kv_rope, dest_loc, c0)
        except Exception:
            # capability fallback: compiler does not support HBM-source indirect DMA
            path = "staged"
            result = None
    _PATH_CACHE[key] = path

    if path == "hbm_to_hbm_indirect":
        _LAST_CONFIG = {
            "path": path,
            "lnc": _LNC,
            "partitions": _P,
            "tokens_per_partition": nt,
            "idx_head_cols": c0,
            "scatter": "indirect_dma_hbm_to_hbm_per_128_tokens",
        }
    else:
        kb = _choose_kb(nt)
        result = _destindex_staged_kernel(kv_nope, kv_rope, dest_loc, kb, _NBUF)
        _LAST_CONFIG = {
            "path": "staged",
            "lnc": _LNC,
            "partitions": _P,
            "tokens_per_partition": nt,
            "nope_tokens_per_partition_per_load": kb,
            "nope_ring_buffers": _NBUF,
            "scatter": "indirect_dma_vector_offset_per_128_tokens",
        }

    out_nope, out_rope = result
    return out_nope, out_rope


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
