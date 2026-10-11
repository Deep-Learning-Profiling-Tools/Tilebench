import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration constants (shape-dependent values derived in _plan()).
# ---------------------------------------------------------------------------
_P = 128
_NPROG = 2            # LNC=2: two physical cores per logical core
_W_MAX = 6144         # max columns per contiguous chunk (elements)
_E_MAX = 12288        # max fp16 elements per partition per SBUF tile (24 KiB)
_T_MIN = 8            # min tiles per program when picking rows-per-partition
_R_CAP = 4            # rows per partition <= 4 -> partition groups at 0/32/64/96
_ACT_PERIOD = 4       # every 4th 128-col block on ScalarE
_ACT_MIN_BLOCKS = 64  # only split onto ScalarE when there is enough work

_LAST_CONFIG = {}


@nki.jit
def _weight_dequant_kernel(X, S, rmax, ncw, act_period):
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

    tab32 = tab
    if rmax == 1:
        tab32 = nl.ndarray((128, rtp * NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tab32, src=tab, engine=nisa.engine.vector)

    blk = 0
    done = 0
    while done < rtp:
        lim = min(rtp - done, rmax)
        R = 1
        while R * 2 <= lim:
            R = R * 2
        Q = 128 // R
        rt0 = rt_base + done

        # fp32 per-partition scales: partition p -> S[rt0 + p // Q, :]
        sc = tab32
        sc_off = done * NB
        if rmax > 1:
            sc = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.sbuf)
            for rl in range(R):
                nisa.tensor_copy(
                    dst=sc[rl * Q:(rl + 1) * Q, :],
                    src=tab[rl * Q:(rl + 1) * Q, (done + rl) * NB:(done + rl + 1) * NB],
                    engine=nisa.engine.vector,
                )
            sc_off = 0

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
                b = sc_off + c0 // 128 + bl
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


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _plan(M, N):
    rtp = (M // _P) // _NPROG
    # contiguous column chunks: full rows when N <= _W_MAX
    c = max(1, (N + _W_MAX - 1) // _W_MAX)
    ncw = _round_up((N + c - 1) // c, _P)
    nch = (N + ncw - 1) // ncw
    # rows per partition: as many as fit while keeping enough tiles
    R = 1
    for cand in (4, 2):
        if (cand <= _R_CAP and cand <= rtp and cand * ncw <= _E_MAX
                and (rtp // cand) * nch >= _T_MIN):
            R = cand
            break
    supers = []
    done = 0
    while done < rtp:
        r = _pow2_floor(min(rtp - done, R))
        supers.append(int(r))
        done += r
    tiles = len(supers) * nch
    NB = (N + _P - 1) // _P
    blocks = rtp * NB
    act_period = _ACT_PERIOD if blocks >= _ACT_MIN_BLOCKS else 0
    return {
        "rows_per_partition": int(R),
        "col_chunk": int(ncw),
        "chunks_per_row_group": int(nch),
        "tile_elems": int(R * ncw),
        "act_period": int(act_period),
        "lnc": _NPROG,
        "row_tiles_per_program": int(rtp),
        "super_tiles": supers,
        "tiles_per_program": int(tiles),
        "scale_load": "single replicated DMA per program",
    }


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    global _LAST_CONFIG
    M = int(X.shape[0])
    N = int(X.shape[1])
    assert int(TILE_SIZE) == _P, "implementation specialised for TILE_SIZE == 128"
    assert M % (_P * _NPROG) == 0 and N % _P == 0
    cfg = _plan(M, N)
    out = _weight_dequant_kernel(
        X, S, cfg["rows_per_partition"], cfg["col_chunk"], cfg["act_period"]
    )
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
