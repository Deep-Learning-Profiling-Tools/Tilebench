"""(f) nl.dynamic_range under the default frontend: warning or error? Plus the
compile-time loop family (range / nl.range / nl.affine_range / nl.sequential_range /
nl.static_range): do they warn, and do loop-carried updates (accumulating into one
SBUF tile) give correct results? Plus `with nl.no_reorder():`.
Python warnings are captured with warnings.catch_warnings; compiler diagnostics
printed on stderr (fd 2) are captured around the call."""

import os
import tempfile
import warnings

import _common as C
from _common import nki, nl, nisa, torch

P = 128


def capture(fn):
    """Run fn(); return (result, python_warnings, stderr_text)."""
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = fn()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    err = tmp.read().decode(errors="replace")
    pyw = [f"{x.category.__name__}: {x.message}" for x in w
           if "TRACE_CACHE" not in str(x.message)]
    lines = [ln for ln in err.splitlines()
             if ("warning" in ln.lower() or "deprecat" in ln.lower())
             and "'is not' operator" not in ln and "TRACE_CACHE" not in ln]
    return res, pyw, lines


@nki.jit
def dyn_int(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    T = x.shape[0] // P
    x3 = x.reshape((T, P, x.shape[1]))
    o3 = out.reshape((T, P, x.shape[1]))
    for i in nl.dynamic_range(T):
        t = nl.ndarray((P, x.shape[1]), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x3[i])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=2.0)
        nisa.dma_copy(dst=o3[i], src=t)
    return out


@nki.jit
def sum_blocks(x, kind):
    # acc = sum over 4 column blocks of x: a loop-carried update of one SBUF tile.
    # `kind` is a Python str kernel argument (compile-time constant, part of the
    # cache key); a closure variable would NOT be (see k_cache_specialization.py).
    out = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray((P, 256), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    acc = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    if kind == "range":
        it = range(4)
    elif kind == "nl.range":
        it = nl.range(4)
    elif kind == "affine_range":
        it = nl.affine_range(4)
    elif kind == "sequential_range":
        it = nl.sequential_range(4)
    else:
        it = nl.static_range(4)
    for j in it:
        nisa.tensor_tensor(dst=acc, data1=acc, data2=xt[:, j * 64:(j + 1) * 64], op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


@nki.jit
def no_reorder_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    with nl.no_reorder():
        nisa.dma_copy(dst=t, src=x)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=3.0)
        nisa.dma_copy(dst=out, src=t)
    return out


torch.manual_seed(6)
x_cpu = torch.randn(256, 128, dtype=torch.float32)
x = C.to_dev(x_cpu)


def dyn_case():
    def run():
        out = dyn_int(x)
        C.sync()
        return out
    out, pyw, diag = capture(run)
    ok, err = C.allclose(out, 2 * x_cpu, 1e-6, 1e-6)
    return {"ok": ok, "max_abs_err": err, "python_warnings": pyw, "compiler_warnings": diag}


s_cpu = torch.randn(128, 256, dtype=torch.float32)
s_ref = sum(s_cpu[:, j * 64:(j + 1) * 64] for j in range(4))
s_dev = C.to_dev(s_cpu)


def range_case(kind):
    def f():
        def run():
            out = sum_blocks(s_dev, kind)
            C.sync()
            return out
        out, pyw, diag = capture(run)
        ok, err = C.allclose(out, s_ref, 1e-5, 1e-5)
        return {"ok": ok, "max_abs_err": err, "python_warnings": pyw, "compiler_warnings": diag}
    return f


def no_reorder_case():
    xs = torch.randn(128, 64)
    out = no_reorder_kernel(C.to_dev(xs))
    C.sync()
    ok, err = C.allclose(out, xs + 3.0, 1e-6, 1e-6)
    return {"ok": ok, "max_abs_err": err}


C.run_case("dynamic_range_int_bound", "for i in nl.dynamic_range(T)", "accepted or rejected by the default (parser) frontend; warnings recorded", dyn_case)
for kind in ["range", "nl.range", "affine_range", "sequential_range", "static_range"]:
    C.run_case(f"loop_carried_{kind}", f"{kind}(4)", "loop-carried accumulation into one SBUF tile is correct; warnings recorded", range_case(kind))
C.run_case("no_reorder_block", "with nl.no_reorder():", "context manager accepted; result correct", no_reorder_case)
C.emit()
