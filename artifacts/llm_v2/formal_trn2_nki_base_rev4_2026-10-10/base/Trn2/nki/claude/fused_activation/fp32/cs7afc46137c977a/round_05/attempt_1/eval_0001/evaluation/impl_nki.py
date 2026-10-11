import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fused gated SiLU: out = silu(x * gate + bias), fp32, single streaming pass.
#
# Mapping:
#   - The index range is split into 2 contiguous regions, one per physical
#     core (LNC=2 programs).
#   - Each region is a sequence of tiles. A tile of width w is one contiguous
#     block of 128*w fp32 values, read as a [128, w] SBUF tile (each
#     partition reads w contiguous elements).
#   - Main width F = clamp(L/8, 512, 2048) (L = per-partition elements of a
#     region), as in the best previous variant. Only the LAST F-tile of each
#     region is replaced by a halving ramp (F/2, F/4, ..., >=256, repeated
#     once) so that the drain after the final load (compute + store of the
#     last tile, with HBM otherwise idle) is short. No ramp-up at the start
#     (HBM is kept busy by loads there anyway) and no explicit prefetch
#     ordering: the compiler schedules the unrolled loop freely.
#   - Per tile: VectorE z = x*gate, z = z + bias (fp32, in place);
#     ScalarE z = silu(z) (fp32, in place); DMA z -> out.
#     z never leaves SBUF; only the output is written to HBM.

_P = 128
_NREG = 2
_FMIN = 512
_FMAX = 2048
_TARGET_TILES = 8
_WMIN = 256

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
    T = len(widths)

    for rr in range(rpp):
        base = (pid * rpp + rr) * region
        for i in range(T):
            w = widths[i]
            blk = P * w
            nb = n // blk
            k = (base + offs[i]) // blk
            xt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            gt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            bt = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x.reshape((nb, P, w))[k])
            nisa.dma_copy(dst=gt, src=gate.reshape((nb, P, w))[k])
            nisa.dma_copy(dst=bt, src=bias.reshape((nb, P, w))[k])
            # z = x * gate + bias (fp32, VectorE)
            nisa.tensor_tensor(dst=xt, data1=xt, data2=gt, op=nl.multiply)
            nisa.tensor_tensor(dst=xt, data1=xt, data2=bt, op=nl.add)
            # out = silu(z) (fp32, ScalarE)
            nisa.activation(dst=xt, op=nl.silu, data=xt)
            nisa.dma_copy(dst=out.reshape((nb, P, w))[k], src=xt)

    return out


def _ramp_down(F):
    # F/2, F/4, ..., wmin, wmin  (sums to F); [F] if no ramp possible.
    lst = []
    w = F
    while w % 2 == 0 and (w // 2) >= _WMIN:
        w //= 2
        lst.append(w)
    if len(lst) == 0:
        return [F]
    lst.append(w)
    return lst


def _build_schedule(n):
    # Per-region elements per partition.
    L = n // (_P * _NREG)

    F = L // _TARGET_TILES
    if F > _FMAX:
        F = _FMAX
    if F < _FMIN:
        F = _FMIN
    if F > L:
        F = L
    if F < 1:
        F = 1
    while F > 1 and L % F != 0:
        F //= 2

    T0 = L // F
    down = _ramp_down(F)

    widths = []
    for _ in range(T0 - 1):
        widths.append(F)
    widths.extend(down)

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

    return tuple(int(v) for v in widths), tuple(int(v) for v in offs), int(F), int(L), int(T0), list(down)


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

    widths, offs, F, L, T0, down = _build_schedule(n)

    out = fused_silu_kernel(x, gate, bias, widths, offs)

    _LAST_CONFIG = {
        "partition_size": _P,
        "regions": _NREG,
        "main_width": F,
        "main_tiles_before_tail_split": T0,
        "per_partition_elems_per_region": L,
        "tiles_per_region": len(widths),
        "tail_ramp_down": down,
        "vector_ops": "tensor_tensor(mul), tensor_tensor(add)",
        "scalar_op": "activation(silu)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
