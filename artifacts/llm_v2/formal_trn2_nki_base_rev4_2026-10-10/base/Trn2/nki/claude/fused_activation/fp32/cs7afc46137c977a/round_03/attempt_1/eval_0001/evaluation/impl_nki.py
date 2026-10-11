import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fused gated SiLU: out = silu(x * gate + bias), fp32, one streaming pass.
#
# Mapping:
#   - The index range is split into 2 contiguous regions, one per physical
#     core (LNC=2 programs).
#   - Each region is a sequence of load tiles. A load tile of width w is one
#     contiguous block of 128*w fp32 values, read as a [128, w] SBUF tile
#     (each partition reads w contiguous elements).
#   - Load width FL: about 8 tiles per region for small n (512..2048),
#     and 4096 (16 KiB contiguous per partition row, 2 MiB per DMA) once the
#     region holds at least 8 such tiles. For FL = 4096 the last tile of each
#     region is loaded as two 2048-wide tiles (short pipeline drain).
#   - Compute/store granularity: chunks of at most 2048 columns. Each chunk
#     gets its own SBUF result tile z:
#         VectorE: z = x*gate ; z = z + bias      (fp32)
#         ScalarE: z = silu(z)                    (fp32, in place)
#         DMA    : z -> out
#     Inputs are only read by compute (no false write dependencies) and are
#     freed right after the VectorE ops. z never leaves SBUF; only the
#     output is written to HBM.

_P = 128
_NREG = 2            # regions = physical cores of the logical core (LNC=2)
_FMIN = 512
_FMID = 2048         # max width of the small-n schedule, and compute chunk
_FBIG = 4096         # load width for large n
_TARGET_TILES = 8

_LAST_CONFIG = {}


@nki.jit
def fused_silu_kernel(x, gate, bias, widths, offs, chunk):
    n = x.shape[0]
    P = 128
    NREG = 2
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = NREG // nprog          # regions per program (1 under LNC=2)
    region = n // NREG
    nch = len(widths)

    for rr in range(rpp):
        base = (pid * rpp + rr) * region
        for c in range(nch):
            w = widths[c]
            blk = P * w
            nblk = n // blk
            k = (base + offs[c]) // blk

            xv = x.reshape((nblk, P, w))
            gv = gate.reshape((nblk, P, w))
            bv = bias.reshape((nblk, P, w))
            ov = out.reshape((nblk, P, w))

            xt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            gt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            bt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=xv[k])
            nisa.dma_copy(dst=gt, src=gv[k])
            nisa.dma_copy(dst=bt, src=bv[k])

            cw = min(chunk, w)
            nsub = w // cw
            for j in range(nsub):
                c0 = j * cw
                c1 = c0 + cw
                zt = nl.ndarray((P, cw), dtype=nl.float32, buffer=nl.sbuf)
                # z = x * gate + bias (fp32, VectorE)
                nisa.tensor_tensor(dst=zt, data1=xt[:, c0:c1], data2=gt[:, c0:c1],
                                   op=nl.multiply)
                nisa.tensor_tensor(dst=zt, data1=zt, data2=bt[:, c0:c1], op=nl.add)
                # out = silu(z) (fp32, ScalarE)
                nisa.activation(dst=zt, op=nl.silu, data=zt)
                nisa.dma_copy(dst=ov[k, :, c0:c1], src=zt)

    return out


def _build_schedule(n):
    # Per-region elements per partition.
    L = n // (_P * _NREG)

    if L >= _TARGET_TILES * _FBIG and L % _FBIG == 0:
        FL = _FBIG
    else:
        FL = L // _TARGET_TILES
        if FL > _FMID:
            FL = _FMID
        if FL < _FMIN:
            FL = _FMIN
        if FL > L:
            FL = L
        if FL < 1:
            FL = 1
        while FL > 1 and L % FL != 0:
            FL //= 2

    T = L // FL
    widths = []
    if FL > _FMID and FL % 2 == 0 and T >= 2:
        chunk = FL // 2
        for _ in range(T - 1):
            widths.append(FL)
        widths.append(chunk)
        widths.append(chunk)
        tail_split = True
    else:
        chunk = FL if FL <= _FMID else FL
        for _ in range(T):
            widths.append(FL)
        tail_split = False

    offs = []
    o = 0
    for w in widths:
        offs.append(o)
        o += _P * w
    if o != _P * L:
        raise ValueError("internal schedule error")

    region = n // _NREG
    for w, off in zip(widths, offs):
        blk = _P * w
        if n % blk != 0 or region % blk != 0 or off % blk != 0:
            raise ValueError("unsupported shape for chosen tiling")
        cw = min(chunk, w)
        if w % cw != 0:
            raise ValueError("unsupported chunking")

    return tuple(int(v) for v in widths), tuple(int(v) for v in offs), int(chunk), int(FL), int(L), tail_split


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("x, gate and bias must have identical shapes")
    if x.dtype != torch.float32 or gate.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("fp32 inputs required")
    if x.dim() != 1:
        raise ValueError("1-D inputs expected")
    n = x.numel()
    if n % (_P * _NREG) != 0:
        raise ValueError("n must be a multiple of 256")

    widths, offs, chunk, FL, L, tail_split = _build_schedule(n)

    out = fused_silu_kernel(x, gate, bias, widths, offs, chunk)

    _LAST_CONFIG = {
        "partition_size": _P,
        "regions": _NREG,
        "load_width": FL,
        "compute_chunk": chunk,
        "per_partition_elems_per_region": L,
        "load_tiles_per_region": len(widths),
        "tail_split": bool(tail_split),
        "vector_ops": "tensor_tensor(mul), tensor_tensor(add)",
        "scalar_op": "activation(silu)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
