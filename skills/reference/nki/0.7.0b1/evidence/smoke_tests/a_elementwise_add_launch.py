"""(a) Element-wise add of two fp16 [256, 512] HBM tensors with nisa.dma_copy +
nisa.tensor_tensor. Launch forms: bare kernel(...), kernel[2](...), kernel[1](...).
Also: nl.load / nl.store produce the same result (see j_load_store for details)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def add_kernel(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = nl.tile_size.pmax                      # 128
    rows, cols = a.shape
    for i in range(rows // P):                 # plain Python range: unrolled at trace time
        a_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, cols), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=b_t, src=b[i * P:(i + 1) * P, :])
        c_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_t, data1=a_t, data2=b_t, op=nl.add)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=c_t)
    return out


torch.manual_seed(0)
a_cpu = torch.randn(256, 512, dtype=torch.float16)
b_cpu = torch.randn(256, 512, dtype=torch.float16)
ref = (a_cpu.float() + b_cpu.float())
a, b = C.to_dev(a_cpu), C.to_dev(b_cpu)


def bare():
    out = add_kernel(a, b)
    C.sync()
    ok, err = C.allclose(out, ref, rtol=1e-3, atol=1e-3)
    return {"ok": ok, "max_abs_err": err, "out_device": str(out.device), "out_dtype": str(out.dtype),
            "out_shape": list(out.shape)}


def lnc2():
    out = add_kernel[2](a, b)
    C.sync()
    ok, err = C.allclose(out, ref, rtol=1e-3, atol=1e-3)
    return {"ok": ok, "max_abs_err": err}


def lnc1():
    out = add_kernel[1](a, b)
    C.sync()
    ok, err = C.allclose(out, ref, rtol=1e-3, atol=1e-3)
    return {"ok": ok, "max_abs_err": err}


def repeat_call():
    # second call with same shapes/dtypes reuses the compiled kernel
    import time
    t = time.time()
    out = add_kernel(a, b)
    C.sync()
    ok, err = C.allclose(out, ref, rtol=1e-3, atol=1e-3)
    return {"ok": ok, "max_abs_err": err, "second_call_seconds": round(time.time() - t, 3)}


C.run_case("bare_call", "nki.jit kernel(...)", "bare call with neuron tensors; output allocated in nl.shared_hbm and returned; correctness vs CPU", bare)
C.run_case("lnc2_bracket", "kernel[2](...)", "explicit LNC=2 launch gives the same correct result", lnc2)
C.run_case("lnc1_bracket", "kernel[1](...)", "LNC=1 launch inside an LNC=2 process (NEURON_LOGICAL_NC_CONFIG=2)", lnc1)
C.run_case("repeat_bare_call", "nki.jit kernel(...)", "second identical call (compile cache hit)", repeat_call)
C.emit()
