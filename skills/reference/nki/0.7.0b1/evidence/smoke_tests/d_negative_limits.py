"""(d) Negative tests: what the installed compiler rejects on trn2, with the exact error.
- nc_matmul with int8 inputs
- float8_e4m3fn (OCP) torch tensor passed to a kernel, and nl.float8_e4m3fn tiles in nc_matmul
- partition dimension 129 (SBUF tile / instruction)
- PSUM matmul output free size > 512 fp32 (one 2 KiB bank)
- wrong buffer for nc_matmul operands; fp32 x bf16 operand mix
- removed / Beta-1 APIs: nl.arange, nl.par_dim, mask= keyword
Positive controls: a [128, 1024] fp32 PSUM tile (two banks) written by two N=512
matmuls; fp8_e5m2 and legacy fp8_e4m3 tiles produced on chip and used in nc_matmul."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def mm_int8(lhsT, rhs):
    out = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray(lhsT.shape, dtype=nl.int8, buffer=nl.sbuf)
    m = nl.ndarray(rhs.shape, dtype=nl.int8, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT)
    nisa.dma_copy(dst=m, src=rhs)
    acc = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s, moving=m)
    r = nl.ndarray(acc.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def mm_onchip_fp8(lhsT, rhs, fp8_dtype):
    # inputs arrive as bf16; cast on chip to an fp8 dtype, then nc_matmul
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s16 = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m16 = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s16, src=lhsT)
    nisa.dma_copy(dst=m16, src=rhs)
    s8 = nl.ndarray((K, M), dtype=fp8_dtype, buffer=nl.sbuf)
    m8 = nl.ndarray((K, N), dtype=fp8_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s8, src=s16)
    nisa.tensor_copy(dst=m8, src=m16)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s8, moving=m8)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def passthrough(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def add_rows(a, b):
    # one tile covering all rows of a: invalid when a.shape[0] > 128
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    ta = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    tb = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ta, src=a)
    nisa.dma_copy(dst=tb, src=b)
    nisa.tensor_tensor(dst=ta, data1=ta, data2=tb, op=nl.add)
    nisa.dma_copy(dst=out, src=ta)
    return out


@nki.jit
def mm_wide(lhsT, rhs, two_banks):
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT)
    nisa.dma_copy(dst=m, src=rhs)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    if two_banks:
        for j in range(N // 512):
            nisa.nc_matmul(dst=acc[:, j * 512:(j + 1) * 512], stationary=s,
                           moving=m[:, j * 512:(j + 1) * 512])
    else:
        nisa.nc_matmul(dst=acc, stationary=s, moving=m)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def mm_moving_in_psum(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    m = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.psum)
    nisa.dma_copy(dst=s, src=lhsT)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s, moving=m)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def mm_from_hbm(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=lhsT, moving=rhs)   # operands still in HBM
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def mm_mixed_fp32_bf16(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.sbuf)
    m = nl.ndarray((K, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=lhsT)
    nisa.dma_copy(dst=m, src=rhs)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=s, moving=m)
    r = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def uses_arange(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    i_p = nl.arange(128)[:, None]
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[i_p, 0:64])
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def uses_par_dim(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((nl.par_dim(128), 64), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def uses_mask(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.load(x, mask=None)
    nl.store(out, value=t)
    return out


torch.manual_seed(3)


def dev(*shape, dtype=torch.bfloat16):
    return C.to_dev(torch.randn(*shape).to(dtype))


def int8_case():
    lhsT = C.to_dev(torch.randint(-4, 4, (128, 128), dtype=torch.int8))
    rhs = C.to_dev(torch.randint(-4, 4, (128, 512), dtype=torch.int8))
    mm_int8(lhsT, rhs)
    C.sync()


def fp8fn_torch_input():
    x = C.to_dev(torch.randn(128, 64).to(torch.float8_e4m3fn))
    passthrough(x)
    C.sync()


def fp8fn_onchip():
    mm_onchip_fp8(dev(128, 128), dev(128, 512), nl.float8_e4m3fn)
    C.sync()


def fp8_ok(dt):
    def f():
        lhsT_c = torch.randn(128, 128).to(torch.bfloat16)
        rhs_c = torch.randn(128, 512).to(torch.bfloat16)
        out = mm_onchip_fp8(C.to_dev(lhsT_c), C.to_dev(rhs_c), dt)
        C.sync()
        ref = lhsT_c.float().T @ rhs_c.float()
        rel = ((out.cpu() - ref).norm() / ref.norm()).item()
        return {"ok": rel < 0.1, "relative_fro_error_vs_bf16_ref": rel}
    return f


def fp8_e5m2_torch_input():
    # torch.float8_e5m2 maps to nl.float8_e5m2 at the kernel boundary
    x_c = torch.randn(128, 64).to(torch.float8_e5m2)
    out = passthrough(C.to_dev(x_c))
    C.sync()
    same = bool(torch.equal(out.cpu().view(torch.uint8), x_c.view(torch.uint8)))
    return {"ok": same, "out_dtype": str(out.dtype)}


def partition_129():
    add_rows(dev(129, 64), dev(129, 64))
    C.sync()


def psum_n_1024_single():
    mm_wide(dev(128, 128), dev(128, 1024), False)
    C.sync()


def psum_two_banks():
    lhsT_c = torch.randn(128, 128).to(torch.bfloat16)
    rhs_c = torch.randn(128, 1024).to(torch.bfloat16)
    out = mm_wide(C.to_dev(lhsT_c), C.to_dev(rhs_c), True)
    C.sync()
    ok, err = C.allclose(out, lhsT_c.float().T @ rhs_c.float(), 2e-2, 8e-2)
    return {"ok": ok, "max_abs_err": err}


def psum_moving():
    mm_moving_in_psum(dev(128, 128), dev(128, 512, dtype=torch.float32))
    C.sync()


def hbm_operands():
    mm_from_hbm(dev(128, 128), dev(128, 512))
    C.sync()


def mixed_fp32_bf16():
    mm_mixed_fp32_bf16(dev(128, 128, dtype=torch.float32), dev(128, 512))
    C.sync()


def removed(fn):
    def f():
        fn(dev(128, 64))
        C.sync()
    return f


C.run_case("nc_matmul_int8", "nisa.nc_matmul int8 operands", "no int8 matmul on trn2 TensorE", int8_case, expect_error=True)
C.run_case("torch_float8_e4m3fn_input", "torch.float8_e4m3fn kernel argument", "OCP e4m3fn tensors cannot be passed on trn2 (without UNSAFE_FP8FNCAST=1)", fp8fn_torch_input, expect_error=True)
C.run_case("nc_matmul_float8_e4m3fn_tiles", "nisa.nc_matmul nl.float8_e4m3fn operands", "OCP fp8 operands rejected on NeuronCore-v3", fp8fn_onchip, expect_error=True)
C.run_case("nc_matmul_float8_e4m3_legacy_tiles", "nisa.nc_matmul nl.float8_e4m3", "legacy fp8_e4m3 operands (cast on chip with tensor_copy) accepted", fp8_ok(nl.float8_e4m3))
C.run_case("nc_matmul_float8_e5m2_tiles", "nisa.nc_matmul nl.float8_e5m2", "fp8_e5m2 operands (cast on chip) accepted", fp8_ok(nl.float8_e5m2))
C.run_case("torch_float8_e5m2_input", "torch.float8_e5m2 kernel argument", "e5m2 tensors pass through a kernel bit-exactly", fp8_e5m2_torch_input)
C.run_case("partition_dim_129", "nl.ndarray((129, 64), buffer=nl.sbuf) + instructions", "partition dim > 128", partition_129, expect_error=True)
C.run_case("psum_matmul_n_1024", "nisa.nc_matmul moving free size 1024", "N > 512 fp32 (one 2 KiB PSUM bank) per nc_matmul", psum_n_1024_single, expect_error=True)
C.run_case("psum_tile_two_banks", "nl.ndarray((128, 1024), fp32, buffer=nl.psum)", "PSUM tile spanning two banks, each nc_matmul writes N=512 slice", psum_two_banks)
C.run_case("nc_matmul_moving_in_psum", "nisa.nc_matmul moving operand in PSUM", "operands must be SBUF", psum_moving, expect_error=True)
C.run_case("nc_matmul_operands_in_hbm", "nisa.nc_matmul HBM operands", "operands must be SBUF", hbm_operands, expect_error=True)
C.run_case("nc_matmul_fp32_x_bf16", "nisa.nc_matmul fp32 stationary + bf16 moving", "fp32/tf32 cannot mix with other dtypes", mixed_fp32_bf16, expect_error=True)
C.run_case("removed_nl_arange", "nl.arange", "absent in 0.7.0b1", removed(uses_arange), expect_error=True)
C.run_case("removed_nl_par_dim", "nl.par_dim", "absent in 0.7.0b1", removed(uses_par_dim), expect_error=True)
C.run_case("removed_mask_kwarg", "nl.load(..., mask=...)", "Beta-1 mask= keyword not accepted", removed(uses_mask), expect_error=True)
C.emit()
