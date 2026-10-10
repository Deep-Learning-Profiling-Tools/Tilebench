"""(e) nl.fori_loop(lower, upper, body_fun, step=1): an on-device (rolled) loop over
row tiles. The body receives the iteration value (NkiScalar for Python-int bounds,
VirtualRegister for register bounds); it is used as a runtime offset into HBM.
Variants of runtime indexing tried:
  ds_step : nl.fori_loop(0, rows, body, step=P); x[nl.ds(i, P), :]
  select  : x.reshape((T, P, N))[i]            (runtime index on an HBM view)
  ap      : x.reshape((T, P, N)).ap(pattern=[[N, P], [1, N]], offset=0,
                                    scalar_offset=i, indirect_dim=0)
  negative: Python arithmetic on the loop value (x[nl.ds(i * P, P), :])
plus a register-bound trip count read from an int32 input tensor, and nl.while_loop."""

import _common as C
from _common import nki, nl, nisa, torch

P = 128


def _tile_io(x, out, src_view, dst_view):
    t = nl.ndarray((P, x.shape[1]), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=src_view)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=dst_view, src=t)


@nki.jit
def loop_ds_step(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    def body(i):                      # i = 0, P, 2P, ... (the row offset itself)
        _tile_io(x, out, x[nl.ds(i, P), :], out[nl.ds(i, P), :])

    nl.fori_loop(0, x.shape[0], body, step=P)
    return out


@nki.jit
def loop_ds_arith(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    def body(i):
        _tile_io(x, out, x[nl.ds(i * P, P), :], out[nl.ds(i * P, P), :])

    nl.fori_loop(0, x.shape[0] // P, body)
    return out


@nki.jit
def loop_select(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    T = x.shape[0] // P
    x3 = x.reshape((T, P, x.shape[1]))
    o3 = out.reshape((T, P, x.shape[1]))

    def body(i):
        _tile_io(x, out, x3[i], o3[i])

    nl.fori_loop(0, T, body)
    return out


@nki.jit
def loop_ap(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    N = x.shape[1]
    T = x.shape[0] // P
    x3 = x.reshape((T, P, N))
    o3 = out.reshape((T, P, N))

    def body(i):
        # offset added = i * stride of dim 0 of the (T, P, N) view = i * P * N elements
        _tile_io(x, out,
                 x3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=i, indirect_dim=0),
                 o3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=i, indirect_dim=0))

    nl.fori_loop(0, T, body)
    return out


@nki.jit
def loop_register_bound(x, n_tiles):
    # trip count read at run time from an int32 HBM tensor of shape (1, 1)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    # make the output deterministic outside the processed tiles
    z = nl.ndarray((P, x.shape[1]), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=z, value=0.0)
    for j in range(x.shape[0] // P):
        nisa.dma_copy(dst=out[j * P:(j + 1) * P, :], src=z)
    cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cnt, src=n_tiles)
    reg = nisa.register_alloc()
    nisa.register_load(reg, cnt)
    T = x.shape[0] // P
    x3 = x.reshape((T, P, x.shape[1]))
    o3 = out.reshape((T, P, x.shape[1]))

    def body(i):                      # i is a VirtualRegister here
        _tile_io(x, out, x3[i], o3[i])

    nl.fori_loop(0, reg, body)
    return out


@nki.jit
def while_countdown(x):
    # acc += x three times; loop condition kept in an SBUF int32 counter
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    acc = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=cnt, value=3)
    reg = nisa.register_alloc()
    nisa.register_load(reg, cnt)

    def body(r):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=xt, op=nl.add)
        nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.add, operand0=-1)
        nisa.register_load(r, cnt)
        return r

    nl.while_loop(reg, body)
    nisa.dma_copy(dst=out, src=acc)
    return out


torch.manual_seed(5)
x_cpu = torch.randn(512, 256, dtype=torch.float32)
x = C.to_dev(x_cpu)
ref = x_cpu + 1.0


def check(kernel):
    def f():
        out = kernel(x)
        C.sync()
        ok, err = C.allclose(out, ref, 1e-5, 1e-5)
        return {"ok": ok, "max_abs_err": err}
    return f


def reg_bound():
    n = C.to_dev(torch.tensor([[2]], dtype=torch.int32))
    out = loop_register_bound(x, n)
    C.sync()
    o = out.cpu()
    ok_first = torch.allclose(o[:256], ref[:256])
    ok_rest = bool((o[256:] == 0).all())
    return {"ok": bool(ok_first and ok_rest), "first_2_tiles_processed": bool(ok_first),
            "remaining_tiles_untouched": ok_rest}


def while_case():
    xs = torch.randn(128, 64, dtype=torch.float32)
    out = while_countdown(C.to_dev(xs))
    C.sync()
    ok, err = C.allclose(out, 3 * xs, 1e-5, 1e-5)
    return {"ok": ok, "max_abs_err": err}


C.run_case("fori_loop_step_nl_ds", "nl.fori_loop(0, rows, body, step=P) + nl.ds(i, P)", "loop value used directly as the runtime start of nl.ds on HBM rows", check(loop_ds_step))
C.run_case("fori_loop_python_arith_on_i", "nl.fori_loop + nl.ds(i * P, P)", "Python arithmetic on the loop value is rejected by the default frontend", check(loop_ds_arith), expect_error=True)
C.run_case("fori_loop_reshape_select", "nl.fori_loop + HBM reshape((T, P, N))[i]", "runtime index on a reshaped HBM view", check(loop_select))
C.run_case("fori_loop_ap_scalar_offset", "nl.fori_loop + (T,P,N)-view .ap(pattern, offset=0, scalar_offset=i, indirect_dim=0)", "explicit access pattern with scalar_offset", check(loop_ap))
C.run_case("fori_loop_register_upper_bound", "nisa.register_alloc/register_load + nl.fori_loop(0, reg, body)", "runtime trip count from an int32 input", reg_bound)
C.run_case("while_loop_countdown", "nl.while_loop(init_reg, body) with register_load", "body returns next condition register; runs 3 times", while_case)
C.emit()
