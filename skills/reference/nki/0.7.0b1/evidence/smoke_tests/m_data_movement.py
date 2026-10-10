"""(m) Data movement details: nisa.dma_transpose (default vs dge_mode=hwdge),
.ap(pattern=..., offset=...) on HBM, HBM .reshape views and strided slices,
negative-index slices, SBUF->SBUF dma_copy, partition-offset slices of SBUF tiles
as compute operands, PSUM -> HBM DMA, writing into a kernel input, and a kernel
that returns nothing (output tensor passed in by the caller)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def dmat(x, mode):
    out = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.sbuf)
    if mode == "hwdge":
        nisa.dma_transpose(dst=t, src=x, dge_mode=nisa.dge_mode.hwdge)
    else:
        nisa.dma_transpose(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def ap_offset(x):
    # rows 128..255 of a [256, F] HBM tensor through an explicit access pattern
    F = x.shape[1]
    out = nl.ndarray((128, F), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((128, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x.ap(pattern=[[F, 128], [1, F]], offset=128 * F))
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def ap_no_pattern(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x.ap(offset=0))
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def views(x):
    # x: [256, F]
    F = x.shape[1]
    out = nl.ndarray((3, 128, F // 2), dtype=x.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray((128, F // 2), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x.reshape((2, 128, F))[1, :, 0:F // 2])      # == x[128:, :F/2]
    nisa.dma_copy(dst=out[0], src=a)
    b = nl.ndarray((128, F // 2), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b, src=x.reshape((128, 2, F))[:, 1, 0:F:2])         # == x[1::2, ::2]
    nisa.dma_copy(dst=out[1], src=b)
    c = nl.ndarray((128, F // 2), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=c, src=x[-128:, F // 2:])                           # negative start index
    nisa.dma_copy(dst=out[2], src=c)
    return out


@nki.jit
def sbuf_to_sbuf_and_partition_slice(x, p0, p1):
    # x: [128, F]; rows p0..p1-1 get +1 via a partition-offset slice of an SBUF tile
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    u = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=u, src=t)                                            # SBUF -> SBUF DMA
    nisa.tensor_scalar(dst=u[p0:p1, :], data=u[p0:p1, :], op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=out, src=u)
    return out


@nki.jit
def dma_into_partition_offset(x):
    # DMA writes rows 5 and 37..39 of an SBUF tile (arbitrary partition starts)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    nisa.dma_copy(dst=t[5:6, :], src=x[0:1, :])
    nisa.dma_copy(dst=t[37:40, :], src=x[1:4, :])
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def psum_to_hbm(x):
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    p = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=p, value=1.0)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def write_into_input(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    nisa.memset(dst=t, value=5.0)
    nisa.dma_copy(dst=x, src=t)            # overwrite the caller's tensor
    return out


@nki.jit
def fill_output_arg(x, out):
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=4.0)
    nisa.dma_copy(dst=out, src=t)


torch.manual_seed(11)


def dmat_case(dt, shape, mode):
    def f():
        x_c = torch.randn(*shape).to(dt)
        out = dmat(C.to_dev(x_c), mode)
        C.sync()
        same = bool(torch.equal(out.cpu(), x_c.T.contiguous()))
        return {"ok": same, "bit_exact": same}
    return f


def ap_case():
    x_c = torch.randn(256, 64)
    out = ap_offset(C.to_dev(x_c))
    C.sync()
    same = bool(torch.equal(out.cpu(), x_c[128:]))
    return {"ok": same}


def ap_missing():
    ap_no_pattern(C.to_dev(torch.randn(128, 64)))
    C.sync()


def views_case():
    x_c = torch.randn(256, 64)
    out = views(C.to_dev(x_c))
    C.sync()
    o = out.cpu()
    checks = {"reshape_select_block": bool(torch.equal(o[0], x_c[128:, :32])),
              "reshape_strided_rows_and_cols": bool(torch.equal(o[1], x_c[1::2, ::2])),
              "negative_index_slice": bool(torch.equal(o[2], x_c[-128:, 32:]))}
    return {"ok": all(checks.values()), **checks}


def part_case(p0, p1):
    def f():
        x_c = torch.zeros(128, 64)
        out = sbuf_to_sbuf_and_partition_slice(C.to_dev(x_c), p0, p1)
        C.sync()
        ref = x_c.clone()
        ref[p0:p1] += 1.0
        same = bool(torch.equal(out.cpu(), ref))
        return {"ok": same}
    return f


def dma_offset_case():
    x_c = torch.randn(128, 64)
    out = dma_into_partition_offset(C.to_dev(x_c))
    C.sync()
    ref = torch.zeros(128, 64)
    ref[5] = x_c[0]
    ref[37:40] = x_c[1:4]
    same = bool(torch.equal(out.cpu(), ref))
    return {"ok": same}


def psum_hbm():
    psum_to_hbm(C.to_dev(torch.zeros(128, 64)))
    C.sync()


def input_write():
    x_c = torch.randn(128, 64)
    x = C.to_dev(x_c)
    out = write_into_input(x)
    C.sync()
    ok_out = bool(torch.equal(out.cpu(), x_c))
    x_after = x.cpu()
    return {"ok": ok_out, "returned_output_correct": ok_out,
            "input_overwritten_on_host_view": bool((x_after == 5.0).all()),
            "input_unchanged": bool(torch.equal(x_after, x_c))}


def no_return():
    x_c = torch.randn(128, 64)
    out = C.to_dev(torch.zeros(128, 64))
    r = fill_output_arg(C.to_dev(x_c), out)
    C.sync()
    filled = bool(torch.equal(out.cpu(), 4 * x_c))
    return {"return_value_type": type(r).__name__, "caller_output_filled": filled}


C.run_case("dma_transpose_bf16_default", "nisa.dma_transpose(dst=sbuf, src=hbm) bf16 [128, 64]", "2-D transpose, default dge_mode", dmat_case(torch.bfloat16, (128, 64), "default"))
C.run_case("dma_transpose_fp32_default", "nisa.dma_transpose fp32 [128, 64]", "default dge_mode accepts fp32", dmat_case(torch.float32, (128, 64), "default"))
C.run_case("dma_transpose_fp32_hwdge", "nisa.dma_transpose(dge_mode=hwdge) fp32", "hwdge requires a 2-byte dtype", dmat_case(torch.float32, (16, 128), "hwdge"), expect_error=True)
C.run_case("dma_transpose_bf16_hwdge_16rows", "nisa.dma_transpose(dge_mode=hwdge) bf16 [16, 128]", "hwdge 2-D: src.shape[0] == 16, src.shape[-1] <= 128", dmat_case(torch.bfloat16, (16, 128), "hwdge"))
C.run_case("ap_pattern_offset", "x.ap(pattern=[[F, 128], [1, F]], offset=128*F)", "explicit HBM access pattern with element offset", ap_case)
C.run_case("ap_without_pattern", "x.ap(offset=0)", "pattern is a required argument", ap_missing, expect_error=True)
C.run_case("hbm_reshape_and_strided_views", "x.reshape(...)[...] / step slices / negative index", "views on HBM tensors as DMA sources", views_case)
for (p0, p1, exp_err) in [(32, 64, False), (0, 100, False), (64, 96, False), (96, 128, False),
                         (8, 24, True), (40, 72, True), (5, 6, True), (33, 34, True)]:
    C.run_case(f"partition_slice_{p0}_{p1}", f"tensor_scalar on SBUF tile rows [{p0}:{p1}] (after SBUF->SBUF dma_copy)",
               "compute-engine access must start at partition 0/32/64/96",
               part_case(p0, p1), expect_error=bool(exp_err))
C.run_case("dma_copy_into_arbitrary_partition_offset", "nisa.dma_copy(dst=t[5:6, :]) / t[37:40, :]", "DMA can address any partition start", dma_offset_case)
C.run_case("dma_copy_psum_to_hbm", "nisa.dma_copy(dst=hbm, src=psum)", "records whether DMA can read PSUM", psum_hbm, expect_error=True)
C.run_case("write_into_kernel_input", "nisa.dma_copy(dst=<input tensor>)", "records whether the caller's tensor is modified", input_write)
C.run_case("kernel_without_return", "kernel(x, out) returning None", "torch-native dispatch needs at least one returned tensor", no_return, expect_error=True)
C.emit()
