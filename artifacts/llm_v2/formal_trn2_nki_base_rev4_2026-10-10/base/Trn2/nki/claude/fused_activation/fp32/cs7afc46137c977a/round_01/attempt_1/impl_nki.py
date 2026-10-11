import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fused gated SiLU: out = silu(x * gate + bias), fp32, single streaming pass.
#
# Mapping:
#   - The 1-D index range is viewed as (ntiles, 128, F): every tile is one
#     contiguous block of 128*F fp32 values. Each partition reads F
#     contiguous elements.
#   - Under LNC=2 the two physical cores (programs) each take a contiguous
#     half of the tiles.
#   - Per tile, the three inputs are DMA'd HBM->SBUF. VectorE computes
#     z = x*gate, then z = z + bias, in place in fp32. ScalarE applies silu
#     in place. The result is DMA'd SBUF->HBM.
#   - z is never written to global memory. Only the output is written.

_P = 128
_NPROG = 2  # bare kernel(...) call runs with LNC=2 on this host

_LAST_CONFIG = {}


@nki.jit
def fused_silu_kernel(x, gate, bias, F):
    n = x.shape[0]
    P = 128
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    ntiles = n // (P * F)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    T = ntiles // nprog

    x3 = x.reshape((ntiles, P, F))
    g3 = gate.reshape((ntiles, P, F))
    b3 = bias.reshape((ntiles, P, F))
    o3 = out.reshape((ntiles, P, F))

    for i in range(T):
        t = pid * T + i
        xt = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        gt = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        bt = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x3[t])
        nisa.dma_copy(dst=gt, src=g3[t])
        nisa.dma_copy(dst=bt, src=b3[t])
        # z = x * gate + bias (fp32, VectorE)
        nisa.tensor_tensor(dst=xt, data1=xt, data2=gt, op=nl.multiply)
        nisa.tensor_tensor(dst=xt, data1=xt, data2=bt, op=nl.add)
        # out = silu(z) (fp32, ScalarE)
        nisa.activation(dst=xt, op=nl.silu, data=xt)
        nisa.dma_copy(dst=o3[t], src=xt)

    return out


def _choose_free_size(n):
    # Per-partition elements handled by each program.
    L = n // (_P * _NPROG)
    # Aim for about 8 tiles per program for small n. Cap the tile free size
    # at 2048 fp32 (8 KiB per partition per stream).
    F = L // 8
    if F > 2048:
        F = 2048
    if F < 512:
        F = 512
    if F > L:
        F = L
    if F < 1:
        F = 1
    while F > 1 and L % F != 0:
        F //= 2
    return F, L


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("x, gate and bias must have identical shapes")
    if x.dtype != torch.float32 or gate.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("fp32 inputs required")
    if x.dim() != 1:
        raise ValueError("1-D inputs expected")
    n = x.numel()
    if n % (_P * _NPROG) != 0:
        raise ValueError("n must be a multiple of 256")

    F, L = _choose_free_size(n)
    ntiles = n // (_P * F)
    if ntiles % _NPROG != 0 or ntiles * _P * F != n:
        raise ValueError("unsupported shape for chosen tiling")

    out = fused_silu_kernel(x, gate, bias, F)

    _LAST_CONFIG = {
        "partition_size": _P,
        "free_size": int(F),
        "num_programs": _NPROG,
        "tiles_total": int(ntiles),
        "tiles_per_program": int(ntiles // _NPROG),
        "per_partition_elems_per_program": int(L),
        "vector_ops": "tensor_tensor(mul), tensor_tensor(add)",
        "scalar_op": "activation(silu)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
