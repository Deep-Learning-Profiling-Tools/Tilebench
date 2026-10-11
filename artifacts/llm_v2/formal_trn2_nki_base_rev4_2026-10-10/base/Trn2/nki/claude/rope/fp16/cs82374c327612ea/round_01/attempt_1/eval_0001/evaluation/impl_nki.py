# Half-split ("rotate-half") RoPE for q (B, S, H, D) with tables cos/sin (S, D/2).
#
# Mapping (all device work in a single NKI kernel, out-of-place):
#   * rows r = b*S + s go on the partition axis, 128 rows per tile; the free axis
#     holds all heads (H*D contiguous values per row). Each tile spans one batch
#     index, because rows_per_tile divides S, so the table row is (r % S).
#   * Tiles are split between the two physical cores of the logical core (LNC=2).
#   * Products are computed in the input dtype, as the reference does:
#       T1 = q * [c, c]          (coefficient tile broadcast over heads, stride 0)
#       T2 = [q2 * (-s), q1 * s]
#     These run on VectorE, with part of T1 on GpSimd.
#   * The final add T1 + T2 is done on TensorE: two identity matmuls accumulate in
#     fp32 PSUM, which is exact for fp16 inputs. ScalarE evacuates PSUM to fp16
#     SBUF with round-to-nearest-even, so the result matches fp16 add semantics.
#   * The next tile's loads are issued before the current tile's compute and store.
import nki
import nki.language as nl
import nki.isa as nisa

_MM_N = 512            # matmul moving free size (max 512)
_PSUM_W = 1024         # PSUM tile width (2 banks of fp32)
_MAX_ROWS = 128        # partition rows per tile
_MAX_CHUNK_COLS = 2048 # columns (heads*D) per compute chunk

_LAST_CONFIG = {}


def _load_tile(q2, cos, sin, qt, cd, sr, r0, s0, pr, nch, W, D2):
    # q rows (chunked DMAs), cos duplicated into both halves, raw sin
    for c in range(nch):
        c0 = c * W
        nisa.dma_copy(dst=qt[:, c0:c0 + W], src=q2[r0:r0 + pr, c0:c0 + W])
    nisa.dma_copy(dst=cd[:, 0:D2], src=cos[s0:s0 + pr, :])
    nisa.dma_copy(dst=cd[:, D2:D2 + D2], src=cos[s0:s0 + pr, :])
    nisa.dma_copy(dst=sr, src=sin[s0:s0 + pr, :])


@nki.jit
def _rope_kernel(q, cos, sin, pr, hc, psum_w, mm_n):
    B = q.shape[0]
    S = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    D2 = D // 2
    R = B * S
    F = H * D
    dt = q.dtype

    out = nl.ndarray((B, S, H, D), dtype=dt, buffer=nl.shared_hbm)
    q2 = q.reshape((R, F))
    o2 = out.reshape((R, F))

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    T = R // pr
    per = (T + nprog - 1) // nprog
    t_begin = min(T, pid * per)
    t_end = min(T, t_begin + per)

    nch = H // hc
    W = hc * D

    # ---- identity matrix (input dtype) for the TensorE accumulation ----
    iota_c = nl.ndarray((pr, pr), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_c, pattern=[[1, pr]], offset=0, channel_multiplier=0)
    iota_p = nl.ndarray((pr, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_p, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    col_f = nl.ndarray((pr, pr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=col_f, src=iota_c)
    row_f = nl.ndarray((pr, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row_f, src=iota_p)
    ident_f = nl.ndarray((pr, pr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident_f, data=col_f, op0=nl.equal, operand0=row_f)
    ident = nl.ndarray((pr, pr), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident, src=ident_f)

    for t in range(t_begin, t_end):
        r0 = t * pr
        if t == t_begin:
            qt = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
            cd = nl.ndarray((pr, D), dtype=dt, buffer=nl.sbuf)
            sr = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
            _load_tile(q2, cos, sin, qt, cd, sr, r0, r0 % S, pr, nch, W, D2)
        else:
            qt = qt_n
            cd = cd_n
            sr = sr_n
        if t + 1 < t_end:
            rn = (t + 1) * pr
            qt_n = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
            cd_n = nl.ndarray((pr, D), dtype=dt, buffer=nl.sbuf)
            sr_n = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
            _load_tile(q2, cos, sin, qt_n, cd_n, sr_n, rn, rn % S, pr, nch, W, D2)

        # -sin
        sn = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sn, data=sr, op0=nl.multiply, operand0=-1.0)

        # coefficient views broadcast over heads (stride 0)
        cd_b = cd.reshape((pr, 1, D)).broadcast(1, hc)
        sn_b = sn.reshape((pr, 1, 1, D2)).broadcast(1, hc)
        sp_b = sr.reshape((pr, 1, 1, D2)).broadcast(1, hc)
        qv3 = qt.reshape((pr, H, D))
        qv4 = qt.reshape((pr, H, 2, D2))

        for c in range(nch):
            h0 = c * hc
            c0 = h0 * D

            # T1 = q * [c, c]
            t1 = nl.ndarray((pr, W), dtype=dt, buffer=nl.sbuf)
            if c % 2 == 1:
                nisa.tensor_tensor(dst=t1.reshape((pr, hc, D)),
                                   data1=qv3[:, h0:h0 + hc, :],
                                   data2=cd_b, op=nl.multiply,
                                   engine=nisa.engine.gpsimd)
            else:
                nisa.tensor_tensor(dst=t1.reshape((pr, hc, D)),
                                   data1=qv3[:, h0:h0 + hc, :],
                                   data2=cd_b, op=nl.multiply,
                                   engine=nisa.engine.vector)

            # T2 = [q2 * (-s), q1 * s]
            t2 = nl.ndarray((pr, W), dtype=dt, buffer=nl.sbuf)
            t2v = t2.reshape((pr, hc, 2, D2))
            nisa.tensor_tensor(dst=t2v[:, :, 0:1, :],
                               data1=qv4[:, h0:h0 + hc, 1:2, :],
                               data2=sn_b, op=nl.multiply,
                               engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=t2v[:, :, 1:2, :],
                               data1=qv4[:, h0:h0 + hc, 0:1, :],
                               data2=sp_b, op=nl.multiply,
                               engine=nisa.engine.vector)

            # out = T1 + T2 via identity matmuls into PSUM, evacuated by ScalarE
            oc = nl.ndarray((pr, W), dtype=dt, buffer=nl.sbuf)
            ng = (W + psum_w - 1) // psum_w
            for g in range(ng):
                g0 = g * psum_w
                gw = min(psum_w, W - g0)
                ps = nl.ndarray((pr, gw), dtype=nl.float32, buffer=nl.psum)
                nj = (gw + mm_n - 1) // mm_n
                for j in range(nj):
                    j0 = j * mm_n
                    jw = min(mm_n, gw - j0)
                    a0 = g0 + j0
                    nisa.nc_matmul(dst=ps[:, j0:j0 + jw], stationary=ident,
                                   moving=t1[:, a0:a0 + jw], accumulate=False)
                    nisa.nc_matmul(dst=ps[:, j0:j0 + jw], stationary=ident,
                                   moving=t2[:, a0:a0 + jw], accumulate=True)
                nisa.activation(dst=oc[:, g0:g0 + gw], op=nl.copy, data=ps)

            nisa.dma_copy(dst=o2[r0:r0 + pr, c0:c0 + W], src=oc)

    return out


def _rows_per_tile(S):
    pr = _MAX_ROWS
    while pr > 1 and S % pr != 0:
        pr //= 2
    return pr


def _heads_per_chunk(H, D):
    for cand in (16, 8, 4, 2, 1):
        if H % cand == 0 and cand * D <= _MAX_CHUNK_COLS:
            return cand
    return 1


def run(q, cos, sin):
    global _LAST_CONFIG
    B = int(q.shape[0])
    S = int(q.shape[1])
    H = int(q.shape[2])
    D = int(q.shape[3])
    pr = _rows_per_tile(S)
    hc = _heads_per_chunk(H, D)
    _LAST_CONFIG = {
        "rows_per_tile": pr,
        "heads_per_chunk": hc,
        "chunk_cols": hc * D,
        "psum_width": _PSUM_W,
        "matmul_n": _MM_N,
        "lnc_split": "tiles split across 2 programs",
        "t1_engine": "vector (even chunks) / gpsimd (odd chunks)",
        "t2_engine": "vector",
        "combine": "tensor_engine identity matmul accumulate in fp32 psum, scalar engine evacuation",
        "prefetch_depth": 1,
    }
    out = _rope_kernel(q, cos, sin, pr, hc, _PSUM_W, _MM_N)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
