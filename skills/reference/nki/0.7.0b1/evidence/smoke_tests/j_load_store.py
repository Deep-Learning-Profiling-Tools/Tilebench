"""(j) nl.load / nl.store in 0.7.0b1: they exist and are thin wrappers
(nl.load(src, dtype=None) allocates an SBUF tile of src's shape and issues
nisa.dma_copy; nl.store(dst, value) issues nisa.dma_copy). Checked: tiled use on a
[256, 512] tensor, dtype= conversion on load, nl.load_transpose2d, and the error
when the loaded slice has more than 128 rows."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def scale_tiles(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range(x.shape[0] // 128):
        t = nl.load(x[i * 128:(i + 1) * 128, :])          # SBUF tile [128, F]
        r = nl.multiply(t, 2.0)
        nl.store(out[i * 128:(i + 1) * 128, :], value=r)
    return out


@nki.jit
def load_as_fp32(x):
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.load(x, dtype=nl.float32)                       # bf16 HBM -> fp32 SBUF
    assert t.dtype == nl.float32
    nl.store(out, t)
    return out


@nki.jit
def load_t2d(x):
    out = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.load_transpose2d(x)
    nl.store(out, t)
    return out


@nki.jit
def load_whole(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.load(x)            # x has 256 rows -> tile with partition dim 256
    nl.store(out, t)
    return out


torch.manual_seed(10)


def tiles():
    x_c = torch.randn(256, 512, dtype=torch.float32)
    out = scale_tiles(C.to_dev(x_c))
    C.sync()
    ok, err = C.allclose(out, 2 * x_c, 1e-6, 1e-6)
    return {"ok": ok, "max_abs_err": err}


def as_fp32():
    x_c = torch.randn(128, 64).to(torch.bfloat16)
    out = load_as_fp32(C.to_dev(x_c))
    C.sync()
    same = bool(torch.equal(out.cpu(), x_c.float()))
    return {"ok": same, "out_dtype": str(out.dtype)}


def t2d(dt, shape):
    def f():
        x_c = torch.randn(*shape).to(dt)
        out = load_t2d(C.to_dev(x_c))
        C.sync()
        same = bool(torch.equal(out.cpu(), x_c.T.contiguous()))
        return {"ok": same, "bit_exact": same}
    return f


def whole():
    load_whole(C.to_dev(torch.randn(256, 64)))
    C.sync()


C.run_case("nl_load_store_tiled", "nl.load(x[slice]) / nl.multiply / nl.store(out[slice], value=)", "row tiles of a [256, 512] tensor", tiles)
C.run_case("nl_load_dtype_conversion", "nl.load(x, dtype=nl.float32)", "bf16 HBM loaded as fp32 SBUF tile (exact)", as_fp32)
C.run_case("nl_load_transpose2d_bf16", "nl.load_transpose2d (dma_transpose)", "bf16 [128, 64] -> [64, 128] SBUF", t2d(torch.bfloat16, (128, 64)))
C.run_case("nl_load_transpose2d_fp32", "nl.load_transpose2d fp32", "default dge_mode: 4-byte dtype accepted (the 2-byte rule is for dge_mode=hwdge, see m_data_movement)", t2d(torch.float32, (128, 64)))
C.run_case("nl_load_256_rows", "nl.load of a [256, 64] HBM tensor", "SBUF tile partition dim > 128 rejected", whole, expect_error=True)
C.emit()
