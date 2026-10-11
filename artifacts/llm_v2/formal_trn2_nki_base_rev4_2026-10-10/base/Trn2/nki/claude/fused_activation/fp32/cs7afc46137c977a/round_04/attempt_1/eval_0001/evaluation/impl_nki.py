import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fused gated SiLU: out = silu(x * gate + bias), fp32, one streaming pass.
#
# Mapping:
#   - The index range is split into 2 contiguous regions, one per physical
#     core (LNC=2 programs).
#   - Each region is a sequence of tiles. A tile of width w is one contiguous
#     block of 128*w fp32 values, read as a [128, w] SBUF tile.
#   - Widths: main width F = clamp(L/8, 512, 2048) (L = per-partition
#     elements of a region). The first and the last F-tile are replaced by
#     halving ramps (e.g. 256,256,512,1024 ... 1024,512,256,256) so the
#     pipeline starts quickly and the drain after the last load is short.
#   - Software pipelining in program order: loads of tile i+2 are issued
#     before the compute/store of tile i.
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

        # ---- prologue: tile 0 ----
        wa = widths[0]
        blka = P * wa
        nba = n // blka
        ka = (base + offs[0]) // blka
        x0 = nl.ndarray((P, wa), dtype=nl.float32, buffer=nl.sbuf)
        g0 = nl.ndarray((P, wa), dtype=nl.float32, buffer=nl.sbuf)
        b0 = nl.ndarray((P, wa), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x0, src=x.reshape((nba, P, wa))[ka])
        nisa.dma_copy(dst=g0, src=gate.reshape((nba, P, wa))[ka])
        nisa.dma_copy(dst=b0, src=bias.reshape((nba, P, wa))[ka])

        # ---- prologue: tile 1 ----
        if T > 1:
            wb = widths[1]
            blkb = P * wb
            nbb = n // blkb
            kb = (base + offs[1]) // blkb
            x1 = nl.ndarray((P, wb), dtype=nl.float32, buffer=nl.sbuf)
            g1 = nl.ndarray((P, wb), dtype=nl.float32, buffer=nl.sbuf)
            b1 = nl.ndarray((P, wb), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x1, src=x.reshape((nbb, P, wb))[kb])
            nisa.dma_copy(dst=g1, src=gate.reshape((nbb, P, wb))[kb])
            nisa.dma_copy(dst=b1, src=bias.reshape((nbb, P, wb))[kb])
        else:
            x1 = x0
            g1 = g0
            b1 = b0

        # ---- steady state ----
        for i in range(T):
            # prefetch tile i+2
            if i + 2 < T:
                wc = widths[i + 2]
                blkc = P * wc
                nbc = n // blkc
                kc = (base + offs[i + 2]) // blkc
                x2 = nl.ndarray((P, wc), dtype=nl.float32, buffer=nl.sbuf)
                g2 = nl.ndarray((P, wc), dtype=nl.float32, buffer=nl.sbuf)
                b2 = nl.ndarray((P, wc), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=x2, src=x.reshape((nbc, P, wc))[kc])
                nisa.dma_copy(dst=g2, src=gate.reshape((nbc, P, wc))[kc])
                nisa.dma_copy(dst=b2, src=bias.reshape((nbc, P, wc))[kc])
            else:
                x2 = x1
                g2 = g1
                b2 = b1

            # compute + store tile i
            w = widths[i]
            blk = P * w
            nb = n // blk
            k = (base + offs[i]) // blk
            # z = x * gate + bias (fp32, VectorE)
            nisa.tensor_tensor(dst=x0, data1=x0, data2=g0, op=nl.multiply)
            nisa.tensor_tensor(dst=x0, data1=x0, data2=b0, op=nl.add)
            # out = silu(z) (fp32, ScalarE)
            nisa.activation(dst=x0, op=nl.silu, data=x0)
            nisa.dma_copy(dst=out.reshape((nb, P, w))[k], src=x0)

            # rotate pipeline registers
            x0 = x1
            g0 = g1
            b0 = b1
            x1 = x2
            g1 = g2
            b1 = b2

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
    up = list(reversed(down))

    widths = []
    if T0 >= 2:
        widths.extend(up)
        for _ in range(T0 - 2):
            widths.append(F)
        widths.extend(down)
    else:
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

    return tuple(int(v) for v in widths), tuple(int(v) for v in offs), int(F), int(L), int(T0)


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

    widths, offs, F, L, T0 = _build_schedule(n)

    out = fused_silu_kernel(x, gate, bias, widths, offs)

    _LAST_CONFIG = {
        "partition_size": _P,
        "regions": _NREG,
        "main_width": F,
        "main_tiles_before_ramps": T0,
        "per_partition_elems_per_region": L,
        "tiles_per_region": len(widths),
        "ramp_up": list(widths[:len(_ramp_down(F))]) if T0 >= 2 else [],
        "ramp_down": list(_ramp_down(F)),
        "prefetch_depth": 2,
        "vector_ops": "tensor_tensor(mul), tensor_tensor(add)",
        "scalar_op": "activation(silu)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
