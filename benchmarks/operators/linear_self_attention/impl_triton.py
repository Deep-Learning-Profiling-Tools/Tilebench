import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


_DEFAULT_KV_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}


_DEFAULT_OUT_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}


_TILE_SHAPES = [
    (bm, bn, bk)
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64]
]


@triton.jit
def _phi(x):

    return tl.where(x > 0, x + 1.0, tl.exp(x))


@triton.jit
def phi_kernel(Y, X, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n_elements)
    tl.store(Y + offs, _phi(x), mask=offs < n_elements)


def _kv_set_block_size_hook(nargs):
    bm = nargs["BLOCK_M"]
    bn = nargs["BLOCK_N"]
    bk = nargs["BLOCK_K"]
    nargs["kt_desc"].block_shape = [bm, bk]
    nargs["vt_desc"].block_shape = [bn, bk]
    nargs["s_desc"].block_shape = [bm, bn]


@triton.jit
def kv_gemm_kernel(
    kt_desc, vt_desc, s_desc,
    M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_dm = pid_m * BLOCK_M
    offs_dn = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(tl.cdiv(M, BLOCK_K)):


        kt = kt_desc.load([offs_dm, k * BLOCK_K])
        vt = vt_desc.load([offs_dn, k * BLOCK_K])
        acc = tl.dot(kt, vt.T, acc, input_precision="tf32")

    s_desc.store([offs_dm, offs_dn], acc.to(s_desc.dtype))


@triton.jit
def z_kernel(
    Z, PhiK,
    M, D,
    stride_km, stride_kd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    k_ptrs = PhiK + offs_m[:, None] * stride_km + offs_d[None, :] * stride_kd
    for m_start in range(0, M, BLOCK_M):
        valid_m = m_start + offs_m < M
        k = tl.load(
            k_ptrs,
            mask=valid_m[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        acc += tl.sum(k, axis=0)
        k_ptrs += BLOCK_M * stride_km

    tl.store(Z + offs_d, acc, mask=offs_d < D)


def _out_set_block_size_hook(nargs):
    bm = nargs["BLOCK_M"]
    bn = nargs["BLOCK_N"]
    bk = nargs["BLOCK_K"]
    nargs["q_desc"].block_shape = [bm, bk]
    nargs["st_desc"].block_shape = [bn, bk]
    nargs["o_desc"].block_shape = [bm, bn]


@triton.jit
def out_gemm_kernel(
    q_desc, st_desc, o_desc, Z,
    M, D, eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in tl.range(tl.cdiv(D, BLOCK_K)):
        q = q_desc.load([offs_m, k * BLOCK_K])
        st = st_desc.load([offs_n, k * BLOCK_K])
        numer = tl.dot(q, st.T, numer, input_precision="tf32")

        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        z = tl.load(Z + offs_k, mask=offs_k < D, other=0.0)
        denom += tl.sum(q * z[None, :], axis=1)

    out = numer / (denom[:, None] + eps)
    o_desc.store([offs_m, offs_n], out.to(o_desc.dtype))


_kv_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=nw,
            num_stages=ns,
            pre_hook=_kv_set_block_size_hook,
        )
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["M", "D"],
    warmup=1,
    rep=3,
)(kv_gemm_kernel)


_out_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=nw,
            num_stages=ns,
            pre_hook=_out_set_block_size_hook,
        )
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["M", "D"],
    warmup=1,
    rep=3,
)(out_gemm_kernel)


def _launch_z(Z, PhiK, M, D):
    z_kernel[(triton.cdiv(D, 32),)](
        Z, PhiK, M, D,
        PhiK.stride(0), PhiK.stride(1),
        BLOCK_M=32,
        BLOCK_D=32,
        num_warps=4,
        num_stages=2,
    )


def run(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    PhiQ = torch.empty_like(Q)
    PhiK = torch.empty_like(K)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)

    n_elements = M * D
    phi_grid = (triton.cdiv(n_elements, 1024),)
    phi_kernel[phi_grid](PhiQ, Q, n_elements, BLOCK=1024, num_warps=8)
    phi_kernel[phi_grid](PhiK, K, n_elements, BLOCK=1024, num_warps=8)


    PhiK_t = PhiK.t().contiguous()
    V_t = V.t().contiguous()

    if autotune:
        dummy = [1, 1]
        kt_desc = TensorDescriptor.from_tensor(PhiK_t, dummy)
        vt_desc = TensorDescriptor.from_tensor(V_t, dummy)
        s_desc = TensorDescriptor.from_tensor(S, dummy)
        kv_grid = lambda meta: (
            triton.cdiv(D, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
        )
        _kv_kernel_autotuned[kv_grid](kt_desc, vt_desc, s_desc, M, D)

        _launch_z(Z, PhiK, M, D)

        S_t = S.t().contiguous()
        q_desc = TensorDescriptor.from_tensor(PhiQ, dummy)
        st_desc = TensorDescriptor.from_tensor(S_t, dummy)
        o_desc = TensorDescriptor.from_tensor(O, dummy)
        out_grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
        )
        _out_kernel_autotuned[out_grid](q_desc, st_desc, o_desc, Z, M, D, float(eps))
    else:
        kv_cfg = _DEFAULT_KV_CONFIG
        kt_desc = TensorDescriptor.from_tensor(
            PhiK_t, [kv_cfg["BLOCK_M"], kv_cfg["BLOCK_K"]])
        vt_desc = TensorDescriptor.from_tensor(
            V_t, [kv_cfg["BLOCK_N"], kv_cfg["BLOCK_K"]])
        s_desc = TensorDescriptor.from_tensor(
            S, [kv_cfg["BLOCK_M"], kv_cfg["BLOCK_N"]])
        kv_grid = (triton.cdiv(D, kv_cfg["BLOCK_M"]), triton.cdiv(D, kv_cfg["BLOCK_N"]))
        kv_gemm_kernel[kv_grid](
            kt_desc, vt_desc, s_desc, M, D,
            BLOCK_M=kv_cfg["BLOCK_M"],
            BLOCK_N=kv_cfg["BLOCK_N"],
            BLOCK_K=kv_cfg["BLOCK_K"],
            num_warps=kv_cfg["num_warps"],
            num_stages=kv_cfg["num_stages"],
        )

        _launch_z(Z, PhiK, M, D)

        out_cfg = _DEFAULT_OUT_CONFIG
        S_t = S.t().contiguous()
        q_desc = TensorDescriptor.from_tensor(
            PhiQ, [out_cfg["BLOCK_M"], out_cfg["BLOCK_K"]])
        st_desc = TensorDescriptor.from_tensor(
            S_t, [out_cfg["BLOCK_N"], out_cfg["BLOCK_K"]])
        o_desc = TensorDescriptor.from_tensor(
            O, [out_cfg["BLOCK_M"], out_cfg["BLOCK_N"]])
        out_grid = (triton.cdiv(M, out_cfg["BLOCK_M"]), triton.cdiv(D, out_cfg["BLOCK_N"]))
        out_gemm_kernel[out_grid](
            q_desc, st_desc, o_desc, Z, M, D, float(eps),
            BLOCK_M=out_cfg["BLOCK_M"],
            BLOCK_N=out_cfg["BLOCK_N"],
            BLOCK_K=out_cfg["BLOCK_K"],
            num_warps=out_cfg["num_warps"],
            num_stages=out_cfg["num_stages"],
        )

    return O


def _cfg_dict(prefix, cfg):
    if cfg is None:
        return {}
    return {
        f"{prefix}_BLOCK_M": cfg.kwargs["BLOCK_M"],
        f"{prefix}_BLOCK_N": cfg.kwargs["BLOCK_N"],
        f"{prefix}_BLOCK_K": cfg.kwargs["BLOCK_K"],
        f"{prefix}_num_warps": cfg.num_warps,
        f"{prefix}_num_stages": cfg.num_stages,
    }


def get_last_config() -> dict | None:
    kv_cfg = getattr(_kv_kernel_autotuned, "best_config", None)
    out_cfg = getattr(_out_kernel_autotuned, "best_config", None)
    if kv_cfg is None or out_cfg is None:
        return None
    return {
        **_cfg_dict("kv", kv_cfg),
        **_cfg_dict("out", out_cfg),
    }
