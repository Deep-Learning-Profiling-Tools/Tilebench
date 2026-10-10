"""(g) Vector / Scalar / GpSimd / Tensor engine instructions (destination-passing style).
- nisa.tensor_scalar with a per-partition [P, 1] tensor operand: fp32 vs fp16 operand
- nisa.activation with vector bias + scalar scale, and the fused add-reduction (reduce_res)
- nisa.tensor_reduce along the free axis (2-D and 3-D, negate=True)
- nisa.nc_transpose on TensorE (PSUM dst dtype == input dtype) and VectorE (32x32)
- cross-partition sum with nc_matmul against a ones vector
- nisa.iota + nisa.tensor_copy cast, nisa.reciprocal, nisa.tensor_tensor_scan
- engine= selection; nl.* convenience ops and the buffer of the tile they allocate."""

import _common as C
from _common import nki, nl, nisa, torch

P = 128


@nki.jit
def ts_vector_operand(x, v):
    # out = (x * v) + 1.0, v is [P, 1] broadcast along the free axis
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    vt = nl.ndarray((x.shape[0], 1), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    nisa.dma_copy(dst=vt, src=v)
    r = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=xt, op0=nl.multiply, operand0=vt, op1=nl.add, operand1=1.0)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def act_bias_scale_reduce(x, b):
    # y = exp(2 * x + b) (b is a [P, 1] vector); s = sum_j y[:, j] via the fused reduction
    y_out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    s_out = nl.ndarray((x.shape[0], 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    bt = nl.ndarray((x.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    nisa.dma_copy(dst=bt, src=b)
    y = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    s = nl.ndarray((x.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y, op=nl.exp, data=xt, bias=bt, scale=2.0,
                    reduce_op=nl.add, reduce_res=s, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.dma_copy(dst=y_out, src=y)
    nisa.dma_copy(dst=s_out, src=s)
    return y_out, s_out


@nki.jit
def act_gpsimd(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    nisa.activation(dst=xt, op=nl.exp, data=xt, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=out, src=xt)
    return out


@nki.jit
def reduce_free(x3):
    # x3: [P, 4, 64] -> max over last axis -> [P, 4]; -sum over axes (1, 2) -> [P, 1]
    m_out = nl.ndarray((x3.shape[0], 4), dtype=nl.float32, buffer=nl.shared_hbm)
    s_out = nl.ndarray((x3.shape[0], 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x3.shape, dtype=x3.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x3)
    m = nl.ndarray((x3.shape[0], 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=xt, axis=2)
    s = nl.ndarray((x3.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, op=nl.add, data=xt, axis=(1, 2), negate=True)
    nisa.dma_copy(dst=m_out, src=m)
    nisa.dma_copy(dst=s_out, src=s)
    return m_out, s_out


@nki.jit
def transpose_pe(x, psum_dtype):
    # x: [128, 64] -> x.T [64, 128] via TensorE (SBUF -> PSUM -> SBUF -> HBM)
    out = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    tp = nl.ndarray((x.shape[1], x.shape[0]), dtype=psum_dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=xt, engine=nisa.engine.tensor)
    ts = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ts, src=tp)
    nisa.dma_copy(dst=out, src=ts)
    return out


@nki.jit
def transpose_ve(x):
    # x: [32, 32] -> x.T on VectorE, SBUF -> SBUF
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    tt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.nc_transpose(dst=tt, data=xt, engine=nisa.engine.vector)
    nisa.dma_copy(dst=out, src=tt)
    return out


@nki.jit
def partition_sum_ones_matmul(x):
    # column sums of x [128, N] (a reduction over the partition axis) = ones[128,1].T @ x
    N = x.shape[1]
    out = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    ones = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    acc = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=ones, moving=xt)
    r = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def iota_cast_recip_scan(dummy):
    # iota: v[p, j] = 1 + 64 * p + j (int32) -> fp32 cast; reciprocal; cumulative sum along free axis
    out_i = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    out_r = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    out_c = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    iv = nl.ndarray((P, 64), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iv, pattern=[[1, 64]], offset=1, channel_multiplier=64)
    fv = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=fv, src=iv)                      # int32 -> fp32 cast
    rv = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rv, data=fv)
    ones = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    cs = nl.ndarray((P, 64), dtype=nl.float32, buffer=nl.sbuf)
    # cs[:, j] = (ones[:, j] * cs[:, j-1]) + fv[:, j], cs[:, -1] := initial = 0
    nisa.tensor_tensor_scan(dst=cs, data0=ones, data1=fv, initial=0.0, op0=nl.multiply, op1=nl.add)
    nisa.dma_copy(dst=out_i, src=fv)
    nisa.dma_copy(dst=out_r, src=rv)
    nisa.dma_copy(dst=out_c, src=cs)
    return out_i, out_r, out_c


@nki.jit
def engines(a, b):
    # same op on explicitly chosen engines
    out = nl.ndarray((3,) + a.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    at = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    bt = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a)
    nisa.dma_copy(dst=bt, src=b)
    r0 = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.sbuf)
    r1 = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.sbuf)
    r2 = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r0, data1=at, data2=bt, op=nl.multiply, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=r1, data1=at, data2=bt, op=nl.multiply, engine=nisa.engine.gpsimd)
    nisa.tensor_copy(dst=r2, src=at, engine=nisa.engine.scalar)
    nisa.dma_copy(dst=out[0], src=r0)
    nisa.dma_copy(dst=out[1], src=r1)
    nisa.dma_copy(dst=out[2], src=r2)
    return out


@nki.jit
def nl_convenience(a, b):
    # nl.* ops allocate and return a new tile; record which buffer they use
    out = nl.ndarray((4,) + a.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    rs_out = nl.ndarray((a.shape[0], 1), dtype=nl.float32, buffer=nl.shared_hbm)
    at = nl.load(a)                                  # SBUF tile, same shape/dtype as a
    bt = nl.load(b)
    s = nl.add(at, bt)                               # tensor_tensor -> new tile in at.buffer
    e = nl.exp(at)                                   # activation -> new tile
    m = nl.multiply(at, 2.0)                         # tensor_scalar -> new tile
    rs = nl.sum(at, axis=1, keepdims=True)           # tensor_reduce -> new SBUF tile [P, 1]
    mx = nl.maximum(at, rs)                          # [P, 1] operand -> tensor_scalar
    assert nl.is_sbuf(s.buffer) and nl.is_sbuf(e.buffer) and nl.is_sbuf(rs.buffer)
    assert s.dtype == at.dtype
    nl.store(out[0], s)
    nl.store(out[1], e)
    nl.store(out[2], m)
    nl.store(out[3], mx)
    nl.store(rs_out, rs)
    return out, rs_out


@nki.jit
def tt_both_psum(x):
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    p1 = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.psum)
    p2 = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=p1, value=1.0)
    nisa.memset(dst=p2, value=2.0)
    r = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r, data1=p1, data2=p2, op=nl.add)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def tt_dst_hbm(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    nisa.tensor_tensor(dst=out, data1=xt, data2=xt, op=nl.add)
    return out


torch.manual_seed(7)


def ts_case(vdt, expect_ok):
    def f():
        x_c = torch.randn(P, 256, dtype=torch.float32)
        v_c = torch.randn(P, 1).to(vdt)
        out = ts_vector_operand(C.to_dev(x_c), C.to_dev(v_c))
        C.sync()
        ok, err = C.allclose(out, x_c * v_c.float() + 1.0, 1e-3, 1e-3)
        return {"ok": ok, "max_abs_err": err}
    return f


def act_case():
    x_c = torch.rand(P, 128, dtype=torch.float32) - 0.5
    b_c = torch.randn(P, 1, dtype=torch.float32) * 0.1
    y, s = act_bias_scale_reduce(C.to_dev(x_c), C.to_dev(b_c))
    C.sync()
    y_ref = torch.exp(2 * x_c + b_c)
    ok1, e1 = C.allclose(y, y_ref, 1e-2, 1e-2)
    ok2, e2 = C.allclose(s, y_ref.sum(dim=1, keepdim=True), 1e-2, 1e-1)
    return {"ok": ok1 and ok2, "exp_max_abs_err": e1, "rowsum_max_abs_err": e2}


def act_gpsimd_case():
    act_gpsimd(C.to_dev(torch.randn(P, 64)))
    C.sync()


def reduce_case():
    x_c = torch.randn(P, 4, 64, dtype=torch.float32)
    m, s = reduce_free(C.to_dev(x_c))
    C.sync()
    ok1, e1 = C.allclose(m, x_c.amax(dim=2), 1e-6, 1e-6)
    ok2, e2 = C.allclose(s, -x_c.sum(dim=(1, 2)).unsqueeze(1), 1e-4, 1e-3)
    return {"ok": ok1 and ok2, "max_err_max": e1, "neg_sum_err": e2}


def tr_pe_case(psum_dt, expect_ok=True):
    def f():
        x_c = torch.randn(P, 64).to(torch.bfloat16)
        out = transpose_pe(C.to_dev(x_c), psum_dt)
        C.sync()
        same = bool(torch.equal(out.cpu(), x_c.T.contiguous()))
        return {"ok": same, "bit_exact": same}
    return f


def tr_ve_case():
    x_c = torch.randn(32, 32, dtype=torch.float32)
    out = transpose_ve(C.to_dev(x_c))
    C.sync()
    same = bool(torch.equal(out.cpu(), x_c.T.contiguous()))
    return {"ok": same, "bit_exact": same}


def tr_ve_too_big():
    transpose_ve(C.to_dev(torch.randn(64, 64)))
    C.sync()


def psum_case():
    x_c = torch.randn(P, 256, dtype=torch.float32)
    out = partition_sum_ones_matmul(C.to_dev(x_c))
    C.sync()
    ok, err = C.allclose(out, x_c.sum(dim=0, keepdim=True), 1e-3, 1e-2)
    return {"ok": ok, "max_abs_err": err}


def iota_case():
    i_o, r_o, c_o = iota_cast_recip_scan(C.to_dev(torch.zeros(1, 1)))
    C.sync()
    ref = (1 + 64 * torch.arange(P).unsqueeze(1) + torch.arange(64).unsqueeze(0)).float()
    ok1 = bool(torch.equal(i_o.cpu(), ref))
    ok2, e2 = C.allclose(r_o, 1.0 / ref, 1e-3, 1e-6)
    ok3, e3 = C.allclose(c_o, torch.cumsum(ref, dim=1), 1e-5, 1e-2)
    return {"ok": ok1 and ok2 and ok3, "iota_exact": ok1, "reciprocal_err": e2, "scan_err": e3}


def engines_case():
    a_c = torch.randn(P, 64)
    b_c = torch.randn(P, 64)
    out = engines(C.to_dev(a_c), C.to_dev(b_c))
    C.sync()
    o = out.cpu()
    ok = torch.allclose(o[0], a_c * b_c, 1e-5, 1e-5) and torch.allclose(o[1], a_c * b_c, 1e-5, 1e-5) \
        and torch.allclose(o[2], a_c, 1e-6, 1e-6)
    return {"ok": bool(ok)}


def nl_case():
    a_c = torch.rand(P, 64) - 0.5
    b_c = torch.rand(P, 64) - 0.5
    out, rs = nl_convenience(C.to_dev(a_c), C.to_dev(b_c))
    C.sync()
    o = out.cpu()
    rs_ref = a_c.sum(dim=1, keepdim=True)
    checks = {
        "add": torch.allclose(o[0], a_c + b_c, 1e-5, 1e-5),
        "exp": torch.allclose(o[1], torch.exp(a_c), 1e-2, 1e-3),
        "multiply_scalar": torch.allclose(o[2], 2 * a_c, 1e-5, 1e-5),
        "maximum_vector": torch.allclose(o[3], torch.maximum(a_c, rs_ref), 1e-4, 1e-4),
        "sum_keepdims": torch.allclose(rs.cpu(), rs_ref, 1e-4, 1e-4),
    }
    return {"ok": all(checks.values()), **{k: bool(v) for k, v in checks.items()},
            "buffers": "nl.add/nl.exp/nl.sum results asserted to be SBUF tiles at trace time"}


def neg(kern, *shape):
    def f():
        kern(C.to_dev(torch.randn(*shape)))
        C.sync()
    return f


C.run_case("tensor_scalar_vector_operand_fp32", "nisa.tensor_scalar(op0, operand0=[P,1] fp32 tile, op1, operand1=scalar)", "per-partition fp32 operand broadcast along free axis", ts_case(torch.float32, True))
C.run_case("tensor_scalar_vector_operand_fp16", "nisa.tensor_scalar operand0=[P,1] fp16 tile", "docstring: tensor operands must be float32 for arithmetic ops", ts_case(torch.float16, False), expect_error=True)
C.run_case("activation_bias_scale_fused_reduce", "nisa.activation(op=nl.exp, bias=[P,1], scale=2.0, reduce_op=nl.add, reduce_res, reduce_cmd=reset_reduce)", "exp(scale*x+bias) and its row sum", act_case)
C.run_case("activation_engine_gpsimd", "nisa.activation(engine=nisa.engine.gpsimd)", "GpSimd activation is NeuronCore-v4+ only", act_gpsimd_case, expect_error=True)
C.run_case("tensor_reduce_free_axes", "nisa.tensor_reduce(axis=2) / axis=(1, 2), negate=True", "reduction over trailing free axes", reduce_case)
C.run_case("nc_transpose_tensor_engine_bf16", "nisa.nc_transpose(engine=tensor), PSUM dst bf16", "bit-exact 128x64 transpose; PSUM dst dtype == input dtype", tr_pe_case(nl.bfloat16))
C.run_case("nc_transpose_tensor_engine_fp32_psum_for_bf16", "nisa.nc_transpose PSUM dst fp32 for bf16 input", "gen3: transpose dst dtype must equal input dtype", tr_pe_case(nl.float32), expect_error=True)
C.run_case("nc_transpose_vector_engine_32x32", "nisa.nc_transpose(engine=vector)", "32x32 SBUF->SBUF transpose", tr_ve_case)
C.run_case("nc_transpose_vector_engine_64x64", "nisa.nc_transpose(engine=vector) 64x64", "VectorE transpose limited to 32x32", tr_ve_too_big, expect_error=True)
C.run_case("partition_axis_sum_via_ones_matmul", "nisa.nc_matmul(stationary=ones[P,1], moving=x)", "sum over the partition axis", psum_case)
C.run_case("iota_cast_reciprocal_scan", "nisa.iota / tensor_copy cast / nisa.reciprocal / nisa.tensor_tensor_scan", "generated pattern, int32->fp32, 1/x, running sum", iota_case)
C.run_case("engine_selection", "engine=nisa.engine.vector|gpsimd|scalar", "tensor_tensor on VectorE and GpSimd, tensor_copy on ScalarE", engines_case)
C.run_case("nl_convenience_ops", "nl.load/nl.add/nl.exp/nl.multiply/nl.sum/nl.maximum/nl.store", "return new SBUF tiles; correct values", nl_case)
C.run_case("tensor_tensor_both_inputs_psum", "nisa.tensor_tensor data1, data2 both PSUM", "not allowed", neg(tt_both_psum, P, 64), expect_error=True)
C.run_case("tensor_tensor_dst_hbm", "nisa.tensor_tensor dst in HBM", "compute engines cannot write HBM", neg(tt_dst_hbm, P, 64), expect_error=True)
C.emit()
