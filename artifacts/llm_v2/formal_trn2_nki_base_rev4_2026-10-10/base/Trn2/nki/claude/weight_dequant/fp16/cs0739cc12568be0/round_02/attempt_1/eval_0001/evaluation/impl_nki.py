import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration constants (shape-dependent values derived in _plan()).
# ---------------------------------------------------------------------------
_P = 128
_E_MAX = 8192        # max fp16 elements per partition per SBUF tile (16 KiB)
_E_MIN = 1024        # min fp16 elements per partition per SBUF tile (2 KiB)
_TARGET_TILES = 16   # aim for about this many tiles per program (pipelining)
_NPROG = 2           # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


@nki.jit
def _weight_dequant_kernel(X, S, tile_elems, rmax, use_act):
    # X: (M, N), S: (M/128, N/128); block side 128.
    M = X.shape[0]
    N = X.shape[1]
    NB = S.shape[1]
    out = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rtp = (M // 128) // n_prog          # row tiles handled by this program
    rt_base = pid * rtp

    # ---- count super tiles (R row tiles each, R power of two) ----
    nsup = 0
    done = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        nsup = nsup + 1
        done = done + R

    # ---- scale tables for all super tiles, loaded up front ----
    # super tile s, partition p -> S[rt0_s + p // Q_s, :] stored at cols [s*NB, (s+1)*NB)
    sc16 = nl.ndarray((128, nsup * NB), dtype=S.dtype, buffer=nl.sbuf)
    sc32 = nl.ndarray((128, nsup * NB), dtype=nl.float32, buffer=nl.sbuf)
    done = 0
    s = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        Q = 128 // R
        rt0 = rt_base + done
        for rl in range(R):
            nisa.dma_copy(
                dst=sc16[rl * Q:(rl + 1) * Q, s * NB:(s + 1) * NB],
                src=S.ap(pattern=[[0, Q], [1, NB]], offset=(rt0 + rl) * NB),
            )
        s = s + 1
        done = done + R
    nisa.tensor_copy(dst=sc32, src=sc16, engine=nisa.engine.vector)

    # ---- main streaming loop ----
    done = 0
    s = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        Q = 128 // R
        rt0 = rt_base + done

        # column chunking (multiples of 128 so chunks align with blocks)
        nch = (R * N + tile_elems - 1) // tile_elems
        ncw = (N + nch - 1) // nch
        ncw = ((ncw + 127) // 128) * 128

        # partition p of this super tile holds rows rt0*128 + p*R + g, g < R
        Xv = X.reshape((M // R, R, N))
        Ov = out.reshape((M // R, R, N))
        p0 = rt0 * Q

        c0 = 0
        while c0 < N:
            w = min(ncw, N - c0)
            xt = nl.ndarray((128, R, w), dtype=X.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=Xv[p0:p0 + 128, :, c0:c0 + w])

            for bl in range(w // 128):
                b = c0 // 128 + bl
                col = s * NB + b
                v = xt[:, :, bl * 128:(bl + 1) * 128]
                on_vec = True
                if use_act != 0:
                    if bl % 4 == 3:
                        on_vec = False
                if on_vec:
                    nisa.tensor_scalar(
                        dst=v,
                        data=v,
                        op0=nl.multiply,
                        operand0=sc32[:, col:col + 1],
                        engine=nisa.engine.vector,
                    )
                else:
                    nisa.activation(
                        dst=v,
                        op=nl.copy,
                        data=v,
                        scale=sc32[:, col:col + 1],
                    )

            nisa.dma_copy(dst=Ov[p0:p0 + 128, :, c0:c0 + w], src=xt)
            c0 = c0 + w
        s = s + 1
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
    E = max(_E_MIN, L // _TARGET_TILES)
    E = min(E, _E_MAX)
    E = min(E, max(512, L // 2))                  # >= 2 tiles per program
    if E >= 1024:
        rmax = _pow2_floor(min(8, max(2, E // 1024)))
    else:
        rmax = 1
    supers = []
    tiles = 0
    done = 0
    while done < rtp:
        R = _pow2_floor(min(rtp - done, rmax))
        nch = (R * N + E - 1) // E
        ncw = (N + nch - 1) // nch
        ncw = ((ncw + 127) // 128) * 128
        cnt = (N + ncw - 1) // ncw
        supers.append([int(R), int(ncw), int(cnt)])
        tiles += cnt
        done += R
    use_act = 1 if tiles >= 8 else 0
    return {
        "tile_elems": int(E),
        "rmax": int(rmax),
        "use_act": int(use_act),
        "act_share": "every 4th 128-col block" if use_act else "none",
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
