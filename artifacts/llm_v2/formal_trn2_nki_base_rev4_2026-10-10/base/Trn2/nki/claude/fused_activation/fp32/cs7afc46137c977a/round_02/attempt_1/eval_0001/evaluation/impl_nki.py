import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fused gated SiLU: out = silu(x * gate + bias), fp32, single streaming pass.
#
# Mapping:
#   - The index range is split into 2 contiguous regions, one per physical
#     core (LNC=2 programs).
#   - Each region is processed as a sequence of chunks. A chunk of width w
#     is one contiguous block of 128*w fp32 elements, read as a [128, w]
#     tile (each partition reads w contiguous elements).
#   - The chunk widths follow a fixed schedule derived from n:
#       head  : W/4, W/4, W/2        (fast ramp-up of the pipeline)
#       middle: W, W, ..., W         (large 1 MiB DMAs, W = 2048)
#       tail  : W/2, W/4, W/8, W/8   (short drain: small last compute)
#     Every chunk offset is a multiple of 128*w, so each chunk is addressed
#     through a reshaped HBM view (nblk, 128, w)[k].
#   - Per chunk: 3 DMA loads, VectorE z = x*gate, z = z + bias (fp32, in
#     place), ScalarE silu (in place), 1 DMA store. z never leaves SBUF.

_P = 128
_NREG = 2      # regions = physical cores of the logical core (LNC=2)
_WMAX = 2048   # main chunk width (fp32 elements per partition)

_LAST_CONFIG = {}


@nki.jit
def fused_silu_kernel(x, gate, bias, widths, offs):
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
            # z = x * gate + bias (fp32, VectorE)
            nisa.tensor_tensor(dst=xt, data1=xt, data2=gt, op=nl.multiply)
            nisa.tensor_tensor(dst=xt, data1=xt, data2=bt, op=nl.add)
            # out = silu(z) (fp32, ScalarE)
            nisa.activation(dst=xt, op=nl.silu, data=xt)
            nisa.dma_copy(dst=ov[k], src=xt)

    return out


def _build_schedule(n):
    # Per-region elements per partition.
    L = n // (_P * _NREG)
    W = _WMAX
    while W > 8 and not (L % W == 0 and L >= 2 * W):
        W //= 2
    widths = []
    if L % W == 0 and L >= 2 * W and W >= 8:
        widths.append(W // 4)
        widths.append(W // 4)
        widths.append(W // 2)
        for _ in range(L // W - 2):
            widths.append(W)
        widths.append(W // 2)
        widths.append(W // 4)
        widths.append(W // 8)
        widths.append(W // 8)
    else:
        widths.append(L)
        W = L
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
    return tuple(widths), tuple(offs), W, L


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

    widths, offs, W, L = _build_schedule(n)

    out = fused_silu_kernel(x, gate, bias, widths, offs)

    _LAST_CONFIG = {
        "partition_size": _P,
        "regions": _NREG,
        "main_width": int(W),
        "per_partition_elems_per_region": int(L),
        "chunks_per_region": len(widths),
        "head_widths": [int(v) for v in widths[:3]],
        "tail_widths": [int(v) for v in widths[-4:]],
        "vector_ops": "tensor_tensor(mul), tensor_tensor(add)",
        "scalar_op": "activation(silu)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
