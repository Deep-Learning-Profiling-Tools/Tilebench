"""(i) dtype conversion: fp32 input -> bf16 output.
 - nisa.tensor_copy(dst=bf16 SBUF tile, src=fp32 SBUF tile) then DMA to a bf16 HBM output
 - nisa.dma_copy(dst=bf16 HBM, src=fp32 SBUF): the DMA engine converts on the fly
 - nisa.activation(op=nl.copy) writing a bf16 tile
Results compared bit-for-bit with torch's CPU .to(torch.bfloat16) (round to nearest even);
also fp32 -> fp16 via tensor_copy and bf16 -> fp32 widening (exact)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def cast_tensor_copy(x, out_dtype):
    out = nl.ndarray(x.shape, dtype=out_dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    c = nl.ndarray(x.shape, dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=c, src=t)
    nisa.dma_copy(dst=out, src=c)
    return out


@nki.jit
def cast_by_dma(x):
    out = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)          # fp32 SBUF -> bf16 HBM
    return out


@nki.jit
def cast_by_activation(x):
    out = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    c = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=c, op=nl.copy, data=t)
    nisa.dma_copy(dst=out, src=c)
    return out


torch.manual_seed(9)
x_c = torch.randn(128, 512, dtype=torch.float32) * 3.0
x = C.to_dev(x_c)
ref_bf16 = x_c.to(torch.bfloat16)


def mismatch(out, ref):
    o = out.cpu()
    n_bad = int((o.float() != ref.float()).sum())
    max_ulp_err = (o.float() - ref.float()).abs().max().item()
    return n_bad, max_ulp_err


def tc_bf16():
    out = cast_tensor_copy(x, nl.bfloat16)
    C.sync()
    n_bad, e = mismatch(out, ref_bf16)
    return {"ok": n_bad == 0, "out_dtype": str(out.dtype), "elements_differing_from_torch_rne": n_bad, "max_abs_diff": e}


def tc_fp16():
    out = cast_tensor_copy(x, nl.float16)
    C.sync()
    n_bad, e = mismatch(out, x_c.to(torch.float16))
    return {"ok": n_bad == 0, "out_dtype": str(out.dtype), "elements_differing_from_torch": n_bad, "max_abs_diff": e}


def dma_bf16():
    out = cast_by_dma(x)
    C.sync()
    n_bad, e = mismatch(out, ref_bf16)
    # report, do not assert, the rounding mode of the DMA conversion
    return {"ok": e <= float(ref_bf16.float().abs().max()) * 2 ** -7, "out_dtype": str(out.dtype),
            "elements_differing_from_torch_rne": n_bad, "max_abs_diff": e}


def act_bf16():
    out = cast_by_activation(x)
    C.sync()
    n_bad, e = mismatch(out, ref_bf16)
    return {"ok": e <= float(ref_bf16.float().abs().max()) * 2 ** -7, "elements_differing_from_torch_rne": n_bad, "max_abs_diff": e}


def widen():
    xb = x_c.to(torch.bfloat16)
    out = cast_tensor_copy(C.to_dev(xb), nl.float32)
    C.sync()
    same = bool(torch.equal(out.cpu(), xb.float()))
    return {"ok": same, "exact": same}


C.run_case("tensor_copy_fp32_to_bf16", "nisa.tensor_copy cast fp32 -> bf16", "bf16 output tensor returned; bits vs torch RNE", tc_bf16)
C.run_case("tensor_copy_fp32_to_fp16", "nisa.tensor_copy cast fp32 -> fp16", "bits vs torch", tc_fp16)
C.run_case("dma_copy_fp32_sbuf_to_bf16_hbm", "nisa.dma_copy with dtype conversion", "DMA converts dtype; rounding recorded", dma_bf16)
C.run_case("activation_copy_to_bf16", "nisa.activation(op=nl.copy) bf16 dst", "ScalarE output cast; rounding recorded", act_bf16)
C.run_case("tensor_copy_bf16_to_fp32", "nisa.tensor_copy cast bf16 -> fp32", "exact widening", widen)
C.emit()
