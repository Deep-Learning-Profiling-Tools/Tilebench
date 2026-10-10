"""(c) nisa.nc_matmul: dst[M, N] (PSUM, fp32) = stationary[K, M].T @ moving[K, N],
K = 256 split into two 128-row K tiles accumulated in PSUM, M = 128, N = 512;
PSUM -> SBUF (nisa.tensor_copy, optional dtype cast) -> HBM. bf16 and fp16 inputs.
The kernel takes the left operand already transposed (lhsT[K, M]) so no on-chip
transpose is needed. Also: accumulate=None (default) auto-accumulation, fp32 inputs,
nl.matmul convenience wrapper, and the Beta-1 call style without dst."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def mm_explicit(lhsT, rhs, out_dtype):
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    KT = 128
    for k in range(K // KT):
        s = nl.ndarray((KT, M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m = nl.ndarray((KT, N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=lhsT[k * KT:(k + 1) * KT, :])
        nisa.dma_copy(dst=m, src=rhs[k * KT:(k + 1) * KT, :])
        nisa.nc_matmul(dst=acc, stationary=s, moving=m, accumulate=(k > 0))
    res = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=acc)          # PSUM -> SBUF (+ cast to out_dtype)
    nisa.dma_copy(dst=out, src=res)
    return out


@nki.jit
def mm_auto(lhsT, rhs):
    # accumulate omitted (None): first write to the PSUM tile overwrites, later ones accumulate
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    for k in range(K // 128):
        s = nl.ndarray((128, M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m = nl.ndarray((128, N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=lhsT[k * 128:(k + 1) * 128, :])
        nisa.dma_copy(dst=m, src=rhs[k * 128:(k + 1) * 128, :])
        nisa.nc_matmul(dst=acc, stationary=s, moving=m)
    res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=acc)
    nisa.dma_copy(dst=out, src=res)
    return out


@nki.jit
def mm_nl_matmul(lhsT, rhs):
    # nl.matmul(x, y, transpose_x=True) allocates its PSUM result with x.dtype
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT)
    nisa.dma_copy(dst=m, src=rhs)
    p = nl.matmul(s, m, transpose_x=True)
    res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=p)
    nisa.dma_copy(dst=out, src=res)
    return out


@nki.jit
def mm_beta1_style(lhsT, rhs):
    # Beta-1 style: result returned instead of dst= (no longer valid)
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT)
    nisa.dma_copy(dst=m, src=rhs)
    p = nisa.nc_matmul(s, m)
    res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=p)
    nisa.dma_copy(dst=out, src=res)
    return out


torch.manual_seed(2)
K, M, N = 256, 128, 512


def make(dt):
    lhsT = torch.randn(K, M, dtype=dt)
    rhs = torch.randn(K, N, dtype=dt)
    ref = lhsT.float().T @ rhs.float()
    return C.to_dev(lhsT), C.to_dev(rhs), ref


def case_explicit(dt, out_dt, tol):
    def f():
        lhsT, rhs, ref = make(dt)
        out = mm_explicit(lhsT, rhs, out_dt)
        C.sync()
        ok, err = C.allclose(out, ref, rtol=tol, atol=tol * 4)
        return {"ok": ok, "max_abs_err": err, "out_dtype": str(out.dtype)}
    return f


def case_auto():
    lhsT, rhs, ref = make(torch.bfloat16)
    out = mm_auto(lhsT, rhs)
    C.sync()
    ok, err = C.allclose(out, ref, rtol=2e-2, atol=8e-2)
    # if only the last K tile were kept the error would be O(10)
    return {"ok": ok, "max_abs_err": err}


def case_nl_matmul(dt):
    def f():
        lhsT = torch.randn(128, 128, dtype=dt)
        rhs = torch.randn(128, 512, dtype=dt)
        ref = lhsT.float().T @ rhs.float()
        out = mm_nl_matmul(C.to_dev(lhsT), C.to_dev(rhs))
        C.sync()
        ok, err = C.allclose(out, ref, rtol=2e-2, atol=8e-2)
        return {"ok": ok, "max_abs_err": err}
    return f


def case_beta1():
    lhsT = C.to_dev(torch.randn(128, 128, dtype=torch.bfloat16))
    rhs = C.to_dev(torch.randn(128, 512, dtype=torch.bfloat16))
    out = mm_beta1_style(lhsT, rhs)
    C.sync()
    return {"ok": True}


C.run_case("bf16_k256_accumulate_explicit_fp32_out", "nisa.nc_matmul(dst, stationary, moving, accumulate=(k > 0))",
           "bf16 inputs, 2 K tiles, fp32 PSUM -> SBUF fp32 -> HBM", case_explicit(torch.bfloat16, nl.float32, 2e-2))
C.run_case("fp16_k256_accumulate_explicit_fp16_out", "nisa.nc_matmul + nisa.tensor_copy cast",
           "fp16 inputs, fp32 PSUM, tensor_copy casts to fp16 on the way to SBUF", case_explicit(torch.float16, nl.float16, 2e-2))
C.run_case("fp32_inputs", "nisa.nc_matmul fp32 inputs", "fp32 stationary/moving accepted on trn2", case_explicit(torch.float32, nl.float32, 2e-2))
C.run_case("bf16_accumulate_none_auto", "nisa.nc_matmul(accumulate=None)",
           "default accumulate=None: compiler overwrites on first write, accumulates afterwards", case_auto)
C.run_case("nl_matmul_fp32_inputs", "nl.matmul(x, y, transpose_x=True)", "fp32 inputs (PSUM tile allocated with x.dtype=fp32)", case_nl_matmul(torch.float32))
C.run_case("nl_matmul_bf16_inputs", "nl.matmul(x, y, transpose_x=True)", "bf16 inputs: PSUM result allocated as bf16, rejected on trn2", case_nl_matmul(torch.bfloat16), expect_error=True)
C.run_case("beta1_style_no_dst", "nisa.nc_matmul(stationary, moving) without dst", "old call style returning a tile", case_beta1, expect_error=True)
C.emit()
