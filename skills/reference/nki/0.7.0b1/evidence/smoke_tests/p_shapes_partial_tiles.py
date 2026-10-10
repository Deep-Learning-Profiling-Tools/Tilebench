"""(p) Shapes that are not multiples of the tile size (remainder tiles handled by
slicing the SBUF tile from partition 0), re-specialisation for a new input shape,
and non-contiguous torch inputs (a transposed view)."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def add_one_any_shape(x):
    R, F = x.shape
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range((R + P - 1) // P):
        r0 = i * P
        rows = min(P, R - r0)                     # Python int at trace time
        t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:rows, :], src=x[r0:r0 + rows, :])
        nisa.tensor_scalar(dst=t[0:rows, :], data=t[0:rows, :], op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=out[r0:r0 + rows, :], src=t[0:rows, :])
    return out


torch.manual_seed(14)


def shape_case(r, f):
    def fn():
        x_c = torch.randn(r, f)
        out = add_one_any_shape(C.to_dev(x_c))
        C.sync()
        ok, err = C.allclose(out, x_c + 1, 1e-6, 1e-6)
        return {"ok": ok, "shape": [r, f], "max_abs_err": err}
    return fn


def noncontig_case():
    base = torch.randn(96, 200)
    xt_dev = C.to_dev(base).t()                   # [200, 96] non-contiguous view
    out = add_one_any_shape(xt_dev)
    C.sync()
    ref = base.t() + 1
    ok, err = C.allclose(out, ref, 1e-6, 1e-6)
    return {"ok": True, "result_matches_logical_view": ok, "max_abs_err": err,
            "is_contiguous_input": xt_dev.is_contiguous()}


C.run_case("remainder_rows_200x300", "slice t[0:rows] for the last partial tile", "R=200 (128 + 72 rows), F=300", shape_case(200, 300))
C.run_case("respecialize_new_shape_77x513", "same kernel, new input shape", "a new shape triggers a new trace/compile; result correct", shape_case(77, 513))
C.run_case("noncontiguous_torch_view_input", "kernel(x.t()) with a non-contiguous neuron tensor", "records whether the kernel sees the logical (transposed) values", noncontig_case)
C.emit()
