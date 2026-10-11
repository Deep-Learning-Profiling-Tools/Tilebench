import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration constants (shape-dependent values derived in _plan()).
# ---------------------------------------------------------------------------
_P = 128
_NPROG = 2           # LNC=2: two physical cores per logical core
_E_MAX = 8192        # max fp16 elements per partition per SBUF tile (16 KiB)
_E_MIN = 1024        # min fp16 elements per partition per SBUF tile (2 KiB)
_TARGET_TILES = 16   # aim for about this many tiles per program (pipelining)
_RUN_TARGET = 2048   # target contiguous HBM run (elements) per partition row
_R_CAP = 4           # rows per partition <= 4  -> partition groups start at 0/32/64/96
_ACT_PERIOD = 4      # every 4th 128-col block on ScalarE (when enough tiles)

_LAST_CONFIG = {}


@nki.jit
def _weight_dequant_kernel(X, S, tile_elems, rmax, act_period):
    # X: (M, N), S: (M/128, N/128); block side 128.
    M = X.shape[0]
    N = X.shape[1]
    NB = S.shape[1]
    out = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rtp = (M // 128) // n_prog          # row tiles handled by this program
    rt_base = pid * rtp

    # ---- one DMA: this program's scale rows replicated into all partitions ----
    # tab[p, r*NB + b] = S[rt_base + r, b] for every partition p
    tab = nl.ndarray((128, rtp * NB), dtype=S.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=tab,
        src=S.ap(pattern=[[0, 128], [1, rtp * NB]], offset=rt_base * NB),
    )

    blk = 0
    done = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        Q = 128 // R
        rt0 = rt_base + done

        # per-super-tile fp32 scales: partition p -> S[rt0 + p // Q, :]
        sc = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.sbuf)
        for rl in range(R):
            nisa.tensor_copy(
                dst=sc[rl * Q:(rl + 1) * Q, :],
                src=tab[rl * Q:(rl + 1) * Q, (done + rl) * NB:(done + rl + 1) * NB],
                engine=nisa.engine.vector,
            )

        # column chunking (multiples of 128 so chunks align with blocks)
        wt = tile_elems // R
        nch = (N + wt - 1) // wt
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
                v = xt[:, :, bl * 128:(bl + 1) * 128]
                on_vec = True
                if act_period > 0:
                    if blk % act_period == act_period - 1:
                        on_vec = False
                if on_vec:
                    nisa.tensor_scalar(
                        dst=v,
                        data=v,
                        op0=nl.multiply,
                        operand0=sc[:, b:b + 1],
                        engine=nisa.engine.vector,
                    )
                else:
                    nisa.activation(
                        dst=v,
                        op=nl.copy,
                        data=v,
                        scale=sc[:, b:b + 1],
                    )
                blk = blk + 1

            nisa.dma_copy(dst=Ov[p0:p0 + 128, :, c0:c0 + w], src=xt)
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
    E = max(_E_MIN, L // _TARGET_TILES)
    E = min(E, _E_MAX)
    E = min(E, max(512, L // 2))                  # >= 2 tiles per program
    rmax = _pow2_floor(max(1, min(_R_CAP, E // _RUN_TARGET)))
    supers = []
    tiles = 0
    done = 0
    while done < rtp:
        R = _pow2_floor(min(rtp - done, rmax))
        wt = E // R
        nch = (N + wt - 1) // wt
        ncw = (N + nch - 1) // nch
        ncw = ((ncw + 127) // 128) * 128
        cnt = (N + ncw - 1) // ncw
        supers.append([int(R), int(ncw), int(cnt)])
        tiles += cnt
        done += R
    act_period = _ACT_PERIOD if tiles >= 8 else 0
    return {
        "tile_elems": int(E),
        "rmax": int(rmax),
        "act_period": int(act_period),
        "lnc": _NPROG,
        "row_tiles_per_program": int(rtp),
        "scale_load": "single replicated DMA per program",
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
    out = _weight_dequant_kernel(X, S, cfg["tile_elems"], cfg["rmax"], cfg["act_period"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
