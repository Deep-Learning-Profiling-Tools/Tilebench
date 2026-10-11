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
_TARGET_TILES = 16   # aim for about this many tiles per program
_RUN_TARGET = 2048   # target contiguous HBM run (elements) per partition row
_R_CAP = 4           # rows per partition <= 4 -> partition groups start at 0/32/64/96
_ACT_PERIOD = 4      # every 4th 128-col block on ScalarE (when enough tiles)
_TAIL_MIN = 256      # smallest column width of the graded tail tiles

_LAST_CONFIG = {}


@nki.jit
def _weight_dequant_kernel(X, S, sched, rmax, wbuf, nbuf, prefetch, act_period):
    # X: (M, N), S: (M/128, N/128); block side 128.
    # sched: flat tuple of (R, d, c0, w) per tile (program-local row tiles).
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

    # ---- explicit rotating tile buffers ----
    b0 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b2 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b3 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b4 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b5 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    b6 = b5
    b7 = b5
    if nbuf > 6:
        b6 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)
    if nbuf > 7:
        b7 = nl.ndarray((128, rmax, wbuf), dtype=X.dtype, buffer=nl.sbuf)

    ntiles = len(sched) // 4
    sc = tab32
    sc_off = 0
    prev_d = -1
    blk = 0
    for it in range(ntiles + prefetch):
        # ---------------- issue load of tile `it` ----------------
        if it < ntiles:
            Rl = sched[4 * it]
            dl = sched[4 * it + 1]
            cl = sched[4 * it + 2]
            wl = sched[4 * it + 3]
            pl = (rt_base + dl) * (128 // Rl)
            Xv = X.reshape((M // Rl, Rl, N))
            sl = it % nbuf
            lbuf = b0
            if sl == 1:
                lbuf = b1
            elif sl == 2:
                lbuf = b2
            elif sl == 3:
                lbuf = b3
            elif sl == 4:
                lbuf = b4
            elif sl == 5:
                lbuf = b5
            elif sl == 6:
                lbuf = b6
            elif sl == 7:
                lbuf = b7
            nisa.dma_copy(dst=lbuf[:, 0:Rl, 0:wl], src=Xv[pl:pl + 128, :, cl:cl + wl])

        # ---------------- compute + store tile `it - prefetch` ----------------
        k = it - prefetch
        if k >= 0:
            R = sched[4 * k]
            d = sched[4 * k + 1]
            c0 = sched[4 * k + 2]
            w = sched[4 * k + 3]
            Q = 128 // R
            p0 = (rt_base + d) * Q
            sk = k % nbuf
            cbuf = b0
            if sk == 1:
                cbuf = b1
            elif sk == 2:
                cbuf = b2
            elif sk == 3:
                cbuf = b3
            elif sk == 4:
                cbuf = b4
            elif sk == 5:
                cbuf = b5
            elif sk == 6:
                cbuf = b6
            elif sk == 7:
                cbuf = b7

            if d != prev_d:
                if rmax == 1:
                    sc = tab32
                    sc_off = d * NB
                else:
                    # partition p -> S[rt_base + d + p // Q, :]
                    sc = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.sbuf)
                    for rl in range(R):
                        nisa.tensor_copy(
                            dst=sc[rl * Q:(rl + 1) * Q, :],
                            src=tab[rl * Q:(rl + 1) * Q, (d + rl) * NB:(d + rl + 1) * NB],
                            engine=nisa.engine.vector,
                        )
                    sc_off = 0
                prev_d = d

            for j in range(w // 128):
                b = c0 // 128 + j + sc_off
                v = cbuf[:, 0:R, j * 128:(j + 1) * 128]
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

            Ov = out.reshape((M // R, R, N))
            nisa.dma_copy(dst=Ov[p0:p0 + 128, :, c0:c0 + w], src=cbuf[:, 0:R, 0:w])
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
    L = rtp * N                                   # elements / partition / program
    E = max(_E_MIN, L // _TARGET_TILES)
    E = min(E, _E_MAX)
    E = min(E, max(512, L // 2))                  # >= 2 tiles per program
    rmax = _pow2_floor(max(1, min(_R_CAP, E // _RUN_TARGET)))

    wt = max(128, E // rmax)
    nch = (N + wt - 1) // wt
    ncw = _round_up((N + nch - 1) // nch, 128)

    chunks = []
    c = 0
    while c < N:
        w = min(ncw, N - c)
        chunks.append((c, w))
        c += w

    supers = []
    done = 0
    while done < rtp:
        R = _pow2_floor(min(rtp - done, rmax))
        supers.append((R, done))
        done += R

    sched = []
    tail = []
    ns = len(supers)
    nc = len(chunks)
    for si in range(ns):
        R, d = supers[si]
        for ci in range(nc):
            c0, w = chunks[ci]
            if si == ns - 1 and ci == nc - 1:
                rem = w
                cc = c0
                while rem >= 2 * _TAIL_MIN:
                    h = _round_up(rem // 2, 128)
                    sched += [int(R), int(d), int(cc), int(h)]
                    tail.append(int(h))
                    cc += h
                    rem -= h
                sched += [int(R), int(d), int(cc), int(rem)]
                tail.append(int(rem))
            else:
                sched += [int(R), int(d), int(c0), int(w)]

    ntiles = len(sched) // 4
    act_period = _ACT_PERIOD if ntiles >= 8 else 0
    if E <= 2048:
        nbuf, prefetch = 8, 6
    else:
        nbuf, prefetch = 6, 4

    cfg = {
        "tile_elems": int(E),
        "rmax": int(rmax),
        "wbuf": int(ncw),
        "nbuf": int(nbuf),
        "prefetch": int(prefetch),
        "act_period": int(act_period),
        "lnc": _NPROG,
        "row_tiles_per_program": int(rtp),
        "super_tiles": int(ns),
        "chunks_per_row_tile": int(nc),
        "tail_widths": tail,
        "tiles_per_program": int(ntiles),
        "scale_load": "single replicated DMA per program",
        "pipelining": "explicit rotating buffers, loads issued ahead",
    }
    return cfg, tuple(sched)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    global _LAST_CONFIG
    M = int(X.shape[0])
    N = int(X.shape[1])
    assert int(TILE_SIZE) == _P, "implementation specialised for TILE_SIZE == 128"
    assert M % (_P * _NPROG) == 0 and N % _P == 0
    cfg, sched = _plan(M, N)
    out = _weight_dequant_kernel(
        X, S, sched,
        cfg["rmax"], cfg["wbuf"], cfg["nbuf"], cfg["prefetch"], cfg["act_period"],
    )
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
