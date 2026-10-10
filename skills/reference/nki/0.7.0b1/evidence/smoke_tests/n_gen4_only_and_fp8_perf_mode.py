"""(n) NeuronCore-v4-only APIs on trn2 (NeuronCore-v3): nisa.exponential,
nisa.activate2, nisa.quantize_mx, nisa.dma_copy(priority=...). Expected: rejected.
Positive on trn2: nc_matmul perf_mode=matmul_perf_mode.double_row with fp8 operands
laid out [128, 2, X] (K = 256 split as partition x 2), and nisa.topk availability
(NeuronCore-v3+; only checks that a minimal call compiles and runs)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def k_exponential(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.exponential(dst=t, src=t, max_value=0.0)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def k_activate2(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.activate2(dst=t, op=nl.exp, data=t, imm0=2.0, imm1=0.0, op0=nl.multiply, op1=nl.add)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def k_quantize_mx(x):
    # x: bf16 [32, 64] -> fp8_e5m2_x4 [32, 16] + uint8 scales [32, 16]
    out = nl.ndarray((32, 16), dtype=nl.uint8, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    q = nl.ndarray((32, 16), dtype=nl.float8_e5m2_x4, buffer=nl.sbuf)
    sc = nl.ndarray((32, 16), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=q, src=t, dst_scale=sc)
    nisa.dma_copy(dst=out, src=sc)
    return out


@nki.jit
def k_dma_priority(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x, priority=0)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def k_double_row(lhsT, rhs, fp8_dtype):
    # out[M, N] = lhsT[K, M].T @ rhs[K, N] with K = 256 = 128 partitions x 2
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s16 = nl.ndarray((128, 2, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m16 = nl.ndarray((128, 2, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s16, src=lhsT.reshape((128, 2, M)))     # k = 2 * p + j
    nisa.dma_copy(dst=m16, src=rhs.reshape((128, 2, N)))      # same k mapping
    s8 = nl.ndarray((128, 2, M), dtype=fp8_dtype, buffer=nl.sbuf)
    m8 = nl.ndarray((128, 2, N), dtype=fp8_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s8, src=s16)
    nisa.tensor_copy(dst=m8, src=m16)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s8, moving=m8, perf_mode=nisa.matmul_perf_mode.double_row)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def k_double_row_bf16(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((128, 2, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m = nl.ndarray((128, 2, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT.reshape((128, 2, M)))
    nisa.dma_copy(dst=m, src=rhs.reshape((128, 2, N)))
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s, moving=m, perf_mode=nisa.matmul_perf_mode.double_row)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def k_topk_minimal(x):
    # one 16-partition group, n = 128 bf16 elements in snake layout, k = 8
    vals = nl.ndarray((16, 8), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    t = nl.ndarray((16, 8), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    v = nl.ndarray((16, 8), dtype=nl.bfloat16, buffer=nl.sbuf)
    ix = nl.ndarray((16, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.topk(val_dst=v, idx_dst=ix, src=t, n=128)
    nisa.dma_copy(dst=vals, src=v)
    return vals


torch.manual_seed(12)


def neg(kern, shape, dt=torch.float32):
    def f():
        kern(C.to_dev(torch.randn(*shape).to(dt)))
        C.sync()
    return f


def double_row(dt):
    def f():
        lhsT_c = torch.randn(256, 128).to(torch.bfloat16)
        rhs_c = torch.randn(256, 512).to(torch.bfloat16)
        out = k_double_row(C.to_dev(lhsT_c), C.to_dev(rhs_c), dt)
        C.sync()
        ref = lhsT_c.float().T @ rhs_c.float()
        rel = ((out.cpu() - ref).norm() / ref.norm()).item()
        return {"ok": rel < 0.1, "relative_fro_error_vs_bf16_ref": rel}
    return f


def double_row_bf16():
    k_double_row_bf16(C.to_dev(torch.randn(256, 128).to(torch.bfloat16)),
                      C.to_dev(torch.randn(256, 512).to(torch.bfloat16)))
    C.sync()


def topk_case():
    x_c = torch.randn(16, 8).to(torch.bfloat16)
    out = k_topk_minimal(C.to_dev(x_c))
    C.sync()
    got = sorted(out.cpu().float().flatten().tolist())          # first k results occupy free position 0
    return {"ok": True, "ran_on_trn2": True}


C.run_case("exponential_gen4_only", "nisa.exponential", "NeuronCore-v4+ only", neg(k_exponential, (128, 64)), expect_error=True)
C.run_case("activate2_gen4_only", "nisa.activate2", "NeuronCore-v4+ only", neg(k_activate2, (128, 64)), expect_error=True)
C.run_case("quantize_mx_gen4_only", "nisa.quantize_mx", "NeuronCore-v4+ only (docstring); error recorded", neg(k_quantize_mx, (32, 64), torch.bfloat16), expect_error=True)
C.run_case("dma_copy_priority_gen4_only", "nisa.dma_copy(priority=0)", "DMA QoS priority is NeuronCore-v4+ only", neg(k_dma_priority, (128, 64)), expect_error=True)
C.run_case("nc_matmul_double_row_fp8_e4m3", "nisa.nc_matmul(perf_mode=matmul_perf_mode.double_row) fp8_e4m3", "[128, 2, X] fp8 operands, K = 256 in one instruction", double_row(nl.float8_e4m3))
C.run_case("nc_matmul_double_row_fp8_e5m2", "nisa.nc_matmul(perf_mode=double_row) fp8_e5m2", "same with e5m2", double_row(nl.float8_e5m2))
C.run_case("nc_matmul_double_row_bf16", "nisa.nc_matmul(perf_mode=double_row) bf16", "records whether non-fp8 operands are rejected", double_row_bf16, expect_error=True)
C.run_case("topk_available_trn2", "nisa.topk(val_dst, idx_dst, src, n)", "minimal call compiles and runs on NeuronCore-v3", topk_case)
C.emit()
