import math

import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Static configuration (literals; the per-shape plan is a fixed formula)
# ---------------------------------------------------------------------------
PSUM_FP32_COLS = 512        # one PSUM bank / max matmul moving free size
BLOCK_TARGET_ELEMS = 4096   # target (output rows * padded width) per input block
BLOCK_MIN_ELEMS = 512       # minimum work per block when splitting small problems
MIN_BLOCKS = 4              # aim for at least this many blocks per core when big enough
DMA_PART_GROUP = 32         # partitions per DMA piece
N_PROG_ASSUMED = 2          # LNC=2: two programs (physical cores)

_LAST_CONFIG = {}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _load_block(x, b, r0, nbr, H, W, Wp, RB, pad, kH, C, pg):
    # Padded SBUF buffer: row i <-> input row (r0 - pad + i), col j <-> input col (j - pad).
    xb = nl.ndarray((C, RB, Wp), dtype=x.dtype, buffer=nl.sbuf)
    need = nbr + kH - 1
    top = r0 - pad
    s_lo = max(0, top)
    s_hi = min(H, top + need)
    i_lo = s_lo - top
    i_hi = s_hi - top
    # zero halo rows outside the image
    if i_lo > 0:
        nisa.memset(dst=xb[:, 0:i_lo, :], value=0.0)
    if i_hi < need:
        nisa.memset(dst=xb[:, i_hi:need, :], value=0.0)
    # zero pad columns of the loaded rows
    if pad > 0:
        nisa.memset(dst=xb[:, i_lo:i_hi, 0:pad], value=0.0)
        nisa.memset(dst=xb[:, i_lo:i_hi, pad + W:Wp], value=0.0)
    ng = _ceil_div(C, pg)
    for g in range(ng):
        p0 = g * pg
        p1 = min(C, p0 + pg)
        nisa.dma_copy(dst=xb[p0:p1, i_lo:i_hi, pad:pad + W],
                      src=x[b, p0:p1, s_lo:s_hi, :])
    return xb


@nki.jit
def conv2d_implicit_gemm_kernel(x, w, pad, nb, nr, pg):
    B, C, H, W = x.shape
    Co = w.shape[0]
    Cg = w.shape[1]
    kH = w.shape[2]
    kW = w.shape[3]
    T = kH * kW
    Wp = W + 2 * pad
    OH = H + 2 * pad - kH + 1
    OW = Wp - kW + 1
    RB = nb + kH - 1

    out = nl.ndarray((B, Co, OH, OW), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rows_per = _ceil_div(OH, n_prog)
    row_lo = min(OH, pid * rows_per)
    row_hi = min(OH, row_lo + rows_per)
    nrows = row_hi - row_lo

    # ---- weights: (Co, Cg*T) rows -> per-tap TensorE transpose -> wT[ic, t, oc]
    w_sb = nl.ndarray((Co, Cg * T), dtype=w.dtype, buffer=nl.sbuf)
    w2 = w.reshape((Co, Cg * T))
    for g in range(_ceil_div(Co, pg)):
        p0 = g * pg
        p1 = min(Co, p0 + pg)
        nisa.dma_copy(dst=w_sb[p0:p1, :], src=w2[p0:p1, :])
    w3 = w_sb.reshape((Co, Cg, T))
    wT = nl.ndarray((Cg, T, Co), dtype=w.dtype, buffer=nl.sbuf)
    for kh in range(kH):
        tp = nl.ndarray((Cg, kW, Co), dtype=w.dtype, buffer=nl.psum)
        for kw in range(kW):
            nisa.nc_transpose(dst=tp[:, kw, :], data=w3[:, :, kh * kW + kw],
                              engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=wT[:, kh * kW:(kh + 1) * kW, :], src=tp)

    # ---- implicit GEMM over row blocks
    ev = 0
    if nrows > 0:
        nblk = _ceil_div(nrows, nb)
        for b in range(B):
            cur = _load_block(x, b, row_lo, min(nb, nrows), H, W, Wp, RB, pad, kH, C, pg)
            for bi in range(nblk):
                r0 = row_lo + bi * nb
                nbr = min(nb, row_hi - r0)
                nxt = cur
                if bi + 1 < nblk:
                    r0n = r0 + nb
                    nxt = _load_block(x, b, r0n, min(nb, row_hi - r0n), H, W, Wp, RB,
                                      pad, kH, C, pg)
                xf = cur.reshape((C, RB * Wp))
                ob = nl.ndarray((Co, nb, OW), dtype=x.dtype, buffer=nl.sbuf)
                nch = _ceil_div(nbr, nr)
                base = nbr // nch
                extra = nbr - base * nch
                a = 0
                for ci in range(nch):
                    n = base
                    if ci < extra:
                        n = base + 1
                    L = n * Wp - (kW - 1)
                    ps = nl.ndarray((Co, nr * Wp), dtype=nl.float32, buffer=nl.psum)
                    for t in range(T):
                        kh = t // kW
                        kw = t - kh * kW
                        off = (a + kh) * Wp + kw
                        nisa.nc_matmul(dst=ps[:, 0:L], stationary=wT[:, t, :],
                                       moving=xf[:, off:off + L], accumulate=(t > 0))
                    psv = ps.reshape((Co, nr, Wp))
                    if ev % 2 == 0:
                        nisa.tensor_copy(dst=ob[:, a:a + n, :], src=psv[:, 0:n, 0:OW],
                                         engine=nisa.engine.vector)
                    else:
                        nisa.tensor_copy(dst=ob[:, a:a + n, :], src=psv[:, 0:n, 0:OW],
                                         engine=nisa.engine.scalar)
                    ev = ev + 1
                    a = a + n
                for g in range(_ceil_div(Co, pg)):
                    p0 = g * pg
                    p1 = min(Co, p0 + pg)
                    nisa.dma_copy(dst=out[b, p0:p1, r0:r0 + nbr, :], src=ob[p0:p1, 0:nbr, :])
                cur = nxt
    return out


def _plan(H, W, kH, kW, pad):
    OH = H + 2 * pad - kH + 1
    Wp = W + 2 * pad
    nr = max(1, PSUM_FP32_COLS // Wp)
    rows_per = _ceil_div(OH, N_PROG_ASSUMED)
    work = rows_per * Wp
    nblk = max(_ceil_div(work, BLOCK_TARGET_ELEMS),
               min(MIN_BLOCKS, _ceil_div(work, BLOCK_MIN_ELEMS)))
    nblk = max(1, min(nblk, rows_per))
    nb = _ceil_div(rows_per, nblk)
    nr = max(1, min(nr, nb))
    return nb, nr, nblk


def run(input, weight, stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, C, H, W = [int(s) for s in input.shape]
    Co, Cg, kH, kW = [int(s) for s in weight.shape]
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1:
        raise NotImplementedError("only stride=1, groups=1 supported")
    if C > 128 or Co > 128 or Cg != C:
        raise NotImplementedError("channels must be <= 128")
    Wp = W + 2 * padding
    if Wp > PSUM_FP32_COLS:
        raise NotImplementedError("padded width must be <= 512")

    nb, nr, nblk = _plan(H, W, kH, kW, padding)
    out = conv2d_implicit_gemm_kernel(input, weight, padding, nb, nr, DMA_PART_GROUP)

    _LAST_CONFIG = {
        "algorithm": "implicit_gemm_flat_shift",
        "rows_per_block": nb,
        "rows_per_psum_chunk": nr,
        "blocks_per_core_planned": nblk,
        "psum_cols": nr * Wp,
        "dma_partition_group": DMA_PART_GROUP,
        "n_programs_assumed": N_PROG_ASSUMED,
        "padded_width": Wp,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
