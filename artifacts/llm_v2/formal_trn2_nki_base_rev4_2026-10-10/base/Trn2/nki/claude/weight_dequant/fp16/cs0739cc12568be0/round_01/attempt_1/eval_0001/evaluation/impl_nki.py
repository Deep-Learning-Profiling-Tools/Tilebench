import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration constants (deterministic functions of the shape are computed
# in _plan below).
# ---------------------------------------------------------------------------
_P = 128
_E_MAX = 16384      # max fp16 elements per partition per SBUF tile (32 KiB)
_E_MIN = 2048       # min fp16 elements per partition per SBUF tile (4 KiB)
_NPROG = 2          # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


@nki.jit
def _weight_dequant_kernel(X, S, tile_elems, rmax, use_act):
    # X: (M, N), S: (M/128, N/128); block side = 128 (one SBUF partition span)
    M = X.shape[0]
    N = X.shape[1]
    NB = S.shape[1]
    out = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rt_total = M // 128
    rtp = rt_total // n_prog          # row tiles handled by this program
    rt_base = pid * rtp

    done = 0
    tcount = 0
    while done < rtp:
        # super tile of R row tiles (R power of two, R divides 128)
        rem = rtp - done
        lim = min(rem, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        Q = 128 // R                  # partitions per row tile
        rt0 = rt_base + done          # first global row tile of super tile

        # per-partition scale table: partition p -> S[rt0 + p // Q, :]
        sc16 = nl.ndarray((128, NB), dtype=S.dtype, buffer=nl.sbuf)
        for rl in range(R):
            nisa.dma_copy(
                dst=sc16[rl * Q:(rl + 1) * Q, :],
                src=S.ap(pattern=[[0, Q], [1, NB]], offset=(rt0 + rl) * NB),
            )
        sc32 = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sc32, src=sc16)

        # column chunking (multiples of 128 so chunks align with blocks)
        nch = (R * N + tile_elems - 1) // tile_elems
        ncw = (N + nch - 1) // nch
        ncw = ((ncw + 127) // 128) * 128

        # partition p of this super tile holds rows rt0*128 + p*R + g, g<R
        Xv = X.reshape((M // R, R, N))
        Ov = out.reshape((M // R, R, N))
        p0 = rt0 * Q

        c0 = 0
        while c0 < N:
            w = min(ncw, N - c0)
            xt = nl.ndarray((128, R, w), dtype=X.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=Xv[p0:p0 + 128, :, c0:c0 + w])

            on_vec = True
            if use_act != 0:
                if tcount % 2 == 1:
                    on_vec = False

            for bl in range(w // 128):
                b = c0 // 128 + bl
                v = xt[:, :, bl * 128:(bl + 1) * 128]
                if on_vec:
                    nisa.tensor_scalar(
                        dst=v,
                        data=v,
                        op0=nl.multiply,
                        operand0=sc32[:, b:b + 1],
                        engine=nisa.engine.vector,
                    )
                else:
                    nisa.activation(
                        dst=v,
                        op=nl.copy,
                        data=v,
                        scale=sc32[:, b:b + 1],
                    )

            nisa.dma_copy(dst=Ov[p0:p0 + 128, :, c0:c0 + w], src=xt)
            tcount = tcount + 1
            c0 = c0 + w
        done = done + R
    return out


def _pow2_floor(x):
    r = 1
    while r * 2 <= x:
        r = r * 2
    return r


def _plan(M, N):
    rtp = (M // _P) // _NPROG
    L = rtp * N                                   # elements / partition / program
    E = min(_E_MAX, max(_E_MIN, L // 8))          # aim for >= ~8 tiles per program
    rmax = _pow2_floor(max(2, min(16, E // 1024)))  # keep >= 2 KiB contiguous chunks
    # mirror of the in-kernel decomposition (for config reporting / engine split)
    supers = []
    tiles = 0
    done = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = _pow2_floor(lim)
        nch = (R * N + E - 1) // E
        ncw = (N + nch - 1) // nch
        ncw = ((ncw + 127) // 128) * 128
        c0 = 0
        cnt = 0
        while c0 < N:
            w = min(ncw, N - c0)
            cnt += 1
            c0 += w
        supers.append([R, ncw, cnt])
        tiles += cnt
        done += R
    use_act = 1 if tiles >= 4 else 0
    return {
        "tile_elems": int(E),
        "rmax": int(rmax),
        "use_act": int(use_act),
        "lnc": _NPROG,
        "row_tiles_per_program": int(rtp),
        "super_tiles": supers,
        "tiles_per_program": int(tiles),
    }


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    global _LAST_CONFIG
    M = int(X.shape[0])
    N = int(X.shape[1])
    assert int(TILE_SIZE) == _P, "implementation specialised for TILE_SIZE == 128"
    assert M % (_P * _NPROG) == 0 and N % _P == 0
    cfg = _plan(M, N)
    out = _weight_dequant_kernel(X, S, cfg["tile_elems"], cfg["rmax"], cfg["use_act"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
