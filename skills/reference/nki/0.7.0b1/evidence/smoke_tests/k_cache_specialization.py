"""(k) Compile-time specialisation and the compile caches.

A Python scalar passed as a kernel argument is a compile-time constant and is part
of the cache key (one compile per distinct value). A value captured from an
enclosing function scope (closure / kernel factory) is rejected by the default
(parser) frontend ("unbound variable"). Module-level globals are resolved.
(torch-neuronx also warns that the persistent NKI trace cache, on by default,
"may not capture ... non-argument values"; pass configuration as arguments.)"""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def add_const_arg(x, c):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=c)
    nisa.dma_copy(dst=out, src=t)
    return out


def make_add_const(c):
    @nki.jit
    def add_const_closure(x):
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=c)
        nisa.dma_copy(dst=out, src=t)
        return out
    return add_const_closure


@nki.jit
def add_tuple_bool(x, vals, flag):
    # vals: Python tuple, flag: Python bool -> compile-time constants
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    v = vals[0] + vals[1] if flag else vals[0]
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=float(v))
    nisa.dma_copy(dst=out, src=t)
    return out


SCALE = 3.0   # module-level global read inside a kernel


@nki.jit
def mul_global(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=SCALE)
    nisa.dma_copy(dst=out, src=t)
    return out


xs = torch.zeros(128, 32, dtype=torch.float32)
x = C.to_dev(xs)


def arg_case():
    o1 = add_const_arg(x, 1.0)
    o2 = add_const_arg(x, 2.0)
    C.sync()
    v1, v2 = float(o1.cpu()[0, 0]), float(o2.cpu()[0, 0])
    return {"ok": (v1, v2) == (1.0, 2.0), "c=1.0 ->": v1, "c=2.0 ->": v2}


def tuple_bool_case():
    o1 = add_tuple_bool(x, (1, 2), True)
    o2 = add_tuple_bool(x, (1, 2), False)
    o3 = add_tuple_bool(x, vals=(5, 1), flag=True)     # keyword arguments at the call site
    C.sync()
    v = [float(o.cpu()[0, 0]) for o in (o1, o2, o3)]
    return {"ok": v == [3.0, 1.0, 6.0], "values": v}


def closure_case():
    k1 = make_add_const(1.0)
    o1 = k1(x)
    C.sync()
    return {"value": float(o1.cpu()[0, 0])}


def global_case():
    o = mul_global(C.to_dev(torch.ones(128, 32)))
    C.sync()
    v = float(o.cpu()[0, 0])
    return {"ok": v == 3.0, "value": v}


C.run_case("scalar_kernel_argument", "Python scalar kernel argument", "distinct values give distinct compiled kernels (correct results)", arg_case)
C.run_case("tuple_and_bool_arguments", "Python tuple / bool kernel arguments (positional and keyword)", "compile-time constants usable in Python expressions", tuple_bool_case)
C.run_case("closure_constant_factory", "kernel factory closing over an enclosing-function local", "rejected by the default frontend (unbound variable)", closure_case, expect_error=True)
C.run_case("module_global_constant", "module-level global read in kernel", "resolved at trace time", global_case)
C.emit()
