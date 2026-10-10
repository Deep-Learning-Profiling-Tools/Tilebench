"""(o) Integer data on Vector/GpSimd, the torch <-> NKI dtype mapping at the kernel
boundary (int64, bool), creation helpers (nl.zeros / nl.full / nl.ones / nl.zeros_like /
nl.empty_like) and on-chip capacity errors (PSUM / SBUF over-allocation)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def int_ops(a, b, mode):
    # a, b int32 [128, 64]
    out = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    at = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.sbuf)
    bt = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a)
    nisa.dma_copy(dst=bt, src=b)
    r = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.sbuf)
    if mode == "add_default":
        nisa.tensor_tensor(dst=r, data1=at, data2=bt, op=nl.add)
    elif mode == "add_vector":
        nisa.tensor_tensor(dst=r, data1=at, data2=bt, op=nl.add, engine=nisa.engine.vector)
    elif mode == "bitwise_and_const":
        nisa.tensor_scalar(dst=r, data=at, op0=nl.bitwise_and, operand0=0xFF)
    else:  # right shift by constant
        nisa.tensor_scalar(dst=r, data=at, op0=nl.right_shift, operand0=4)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def compare_mask(a, b):
    out_u8 = nl.ndarray(a.shape, dtype=nl.uint8, buffer=nl.shared_hbm)
    out_f = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    at = nl.load(a)
    bt = nl.load(b)
    m8 = nl.ndarray(a.shape, dtype=nl.uint8, buffer=nl.sbuf)
    mf = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m8, data1=at, data2=bt, op=nl.greater)
    nisa.tensor_tensor(dst=mf, data1=at, data2=bt, op=nl.greater)
    nl.store(out_u8, m8)
    nl.store(out_f, mf)
    return out_u8, out_f


@nki.jit
def passthrough(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.load(x)
    nl.store(out, t)
    return out


@nki.jit
def creation(x):
    out = nl.ndarray((5,) + x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    z = nl.zeros(x.shape, dtype=nl.float32)                 # SBUF by default
    f = nl.full(x.shape, 2.5, dtype=nl.float32)
    o = nl.ones(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    xt = nl.load(x)
    zl = nl.zeros_like(xt)
    e = nl.empty_like(xt)                                    # uninitialised, same shape/dtype/buffer
    assert nl.is_sbuf(e.buffer) and e.shape == xt.shape and e.dtype == xt.dtype
    nisa.tensor_copy(dst=e, src=xt)
    nl.store(out[0], z)
    nl.store(out[1], f)
    nl.store(out[2], o)
    nl.store(out[3], zl)
    nl.store(out[4], e)
    return out


@nki.jit
def zeros_shared_hbm(x):
    out = nl.zeros(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    return out


@nki.jit
def psum_too_big(x):
    out = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    p = nl.ndarray((128, 8192), dtype=nl.float32, buffer=nl.psum)   # 32 KiB / partition > 16 KiB
    nisa.memset(dst=p, value=0.0)
    r = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=p[:, 0:64])
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def sbuf_too_big(x):
    out = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    s = nl.ndarray((128, 60000), dtype=nl.float32, buffer=nl.sbuf)  # 240000 B / partition
    nisa.memset(dst=s, value=0.0)
    nisa.dma_copy(dst=out, src=s[:, 0:64])
    return out


torch.manual_seed(13)
big = 2 ** 30
a_c = torch.randint(-big, big, (128, 64), dtype=torch.int32)
b_c = torch.randint(-big // 2, big // 2, (128, 64), dtype=torch.int32)
a, b = C.to_dev(a_c), C.to_dev(b_c)


def int_case(mode, ref_fn, exact_required, alt_ref_fn=None, alt_name=None):
    def f():
        out = int_ops(a, b, mode)
        C.sync()
        ref = ref_fn()
        o = out.cpu()
        n_bad = int((o != ref).sum())
        rec = {"ok": (n_bad == 0) if exact_required else True, "elements_not_exact": n_bad,
               "out_dtype": str(out.dtype)}
        if alt_ref_fn is not None:
            n_alt = int((o != alt_ref_fn()).sum())
            rec[alt_name + "_mismatches"] = n_alt
            if not exact_required:
                rec["ok"] = n_alt == 0
        return rec
    return f


def logical_shift4():
    u = a_c.to(torch.int64) & 0xFFFFFFFF
    v = u >> 4
    return v.to(torch.int32)


def cmp_case():
    x = torch.randn(128, 64)
    y = torch.randn(128, 64)
    o8, of = compare_mask(C.to_dev(x), C.to_dev(y))
    C.sync()
    ref = (x > y)
    ok = bool(torch.equal(o8.cpu(), ref.to(torch.uint8))) and bool(torch.equal(of.cpu(), ref.float()))
    return {"ok": ok, "uint8_values": sorted(set(o8.cpu().flatten().tolist())),
            "float_values": sorted(set(of.cpu().flatten().tolist()))}


def int64_case():
    x_c = torch.arange(128 * 4, dtype=torch.int64).reshape(128, 4)
    out = passthrough(C.to_dev(x_c))
    C.sync()
    return {"ok": bool(torch.equal(out.cpu().long(), x_c)), "out_dtype": str(out.dtype)}


def bool_case():
    x_c = torch.rand(128, 4) > 0.5
    out = passthrough(C.to_dev(x_c))
    C.sync()
    return {"ok": bool(torch.equal(out.cpu().bool(), x_c)), "out_dtype": str(out.dtype)}


def int8_case():
    x_c = torch.randint(-128, 127, (128, 64), dtype=torch.int8)
    out = passthrough(C.to_dev(x_c))
    C.sync()
    return {"ok": bool(torch.equal(out.cpu(), x_c)), "out_dtype": str(out.dtype)}


def creation_case():
    x_c = torch.randn(128, 64)
    out = creation(C.to_dev(x_c))
    C.sync()
    o = out.cpu()
    checks = {"zeros": bool((o[0] == 0).all()), "full_2.5": bool((o[1] == 2.5).all()),
              "ones": bool((o[2] == 1).all()), "zeros_like": bool((o[3] == 0).all()),
              "empty_like_then_copy": bool(torch.equal(o[4], x_c))}
    return {"ok": all(checks.values()), **checks}


def neg(kern):
    def f():
        kern(C.to_dev(torch.zeros(128, 64)))
        C.sync()
    return f


C.run_case("int32_add_default_engine", "nisa.tensor_tensor int32 add (default engine)", "exact 32-bit integer add (|values| up to 1.5 * 2**30)",
           int_case("add_default", lambda: a_c + b_c, True))
C.run_case("int32_add_vector_engine", "nisa.tensor_tensor int32 add engine=vector", "records exactness when forced onto VectorE (fp32 internal math)",
           int_case("add_vector", lambda: a_c + b_c, False))
C.run_case("int32_bitwise_and_const", "nisa.tensor_scalar(op0=nl.bitwise_and, operand0=0xFF)", "bit-exact", int_case("bitwise_and_const", lambda: a_c & 0xFF, True))
C.run_case("int32_right_shift_const", "nisa.tensor_scalar(op0=nl.right_shift, operand0=4)", "nl.right_shift on int32 is a logical shift (elements_not_exact counts mismatches vs torch arithmetic >>)",
           int_case("right_shift", lambda: a_c >> 4, False, logical_shift4, "logical_shift_ref"))
C.run_case("compare_to_uint8_and_fp32", "nisa.tensor_tensor(op=nl.greater)", "comparison writes 1/0 into uint8 or fp32 dst", cmp_case)
C.run_case("torch_int64_input", "torch.int64 kernel argument", "NKI has no 64-bit types (maps to int32); behaviour recorded", int64_case)
C.run_case("torch_bool_input", "torch.bool kernel argument", "bool maps to uint8 at the boundary", bool_case)
C.run_case("torch_int8_roundtrip", "torch.int8 kernel argument", "int8 data movement is exact", int8_case)
C.run_case("creation_helpers", "nl.zeros / nl.full / nl.ones / nl.zeros_like / nl.empty_like", "SBUF tiles with the expected contents", creation_case)
C.run_case("zeros_in_shared_hbm", "nl.zeros(..., buffer=nl.shared_hbm)", "creating a shared_hbm tensor with an init value is rejected", neg(zeros_shared_hbm), expect_error=True)
C.run_case("psum_over_capacity", "nl.ndarray((128, 8192), fp32, buffer=nl.psum)", "PSUM is 16 KiB per partition", neg(psum_too_big), expect_error=True)
C.run_case("sbuf_over_capacity", "nl.ndarray((128, 60000), fp32, buffer=nl.sbuf)", "usable SBUF per partition is 212984 B on trn2", neg(sbuf_too_big), expect_error=True)
C.emit()
