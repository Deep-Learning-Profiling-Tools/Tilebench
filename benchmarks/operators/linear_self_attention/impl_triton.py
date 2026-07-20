"""Blocked-GEMM Triton implementation of linear self-attention.

The expensive matrix products use the same CTA-level decomposition as the
PyTorch baseline:

  S = phi(K)^T @ V
  O = (phi(Q) @ S) / (phi(Q) @ Z + eps),  Z = sum_m phi(K[m])

Stage 1 and Stage 3 both assign one output tile to each CTA and accumulate the
reduction dimension with ``tl.dot``.  This replaces the old scalar
one-S-element-per-CTA and scalar outer-product loops, which repeatedly reread K,
V, Q, and S.  All three backends use fp32 inputs/outputs with TF32 Tensor-Core
matmul precision; accumulation remains fp32.
"""

import torch
import triton
import triton.language as tl


_DEFAULT_KV_CONFIG = {
    "BLOCK_M": 16,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 1,
}

_DEFAULT_OUT_CONFIG = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 1,
}

# Triton and cuTile use the same (M, N, K) tile-shape search space.  Scheduling
# controls are backend-specific: num_warps here and occupancy in cuTile.
_TILE_SHAPES = [
    (bm, bn, bk)
    for bm in [16, 32]
    for bn in [32, 64]
    for bk in [16, 32]
]


@triton.jit
def _phi(x):
    # ELU(x) + 1: x > 0 -> x + 1, otherwise exp(x).
    return tl.where(x > 0, x + 1.0, tl.exp(x))


@triton.jit
def phi_kernel(Y, X, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n_elements)
    tl.store(Y + offs, _phi(x), mask=offs < n_elements)


@triton.jit
def kv_gemm_kernel(
    S, PhiK, V,
    M, D,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_sm, stride_sd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one [BLOCK_M, BLOCK_N] tile of S = phi(K)^T @ V."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_dm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_ptrs = PhiK + offs_k[:, None] * stride_km + offs_dm[None, :] * stride_kd
    v_ptrs = V + offs_k[:, None] * stride_vm + offs_dn[None, :] * stride_vd

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, M, BLOCK_K):
        valid_k = k_start + offs_k < M
        k = tl.load(
            k_ptrs,
            mask=valid_k[:, None] & (offs_dm[None, :] < D),
            other=0.0,
        )
        v = tl.load(
            v_ptrs,
            mask=valid_k[:, None] & (offs_dn[None, :] < D),
            other=0.0,
        )
        acc = tl.dot(tl.trans(k), v, acc, input_precision="tf32")
        k_ptrs += BLOCK_K * stride_km
        v_ptrs += BLOCK_K * stride_vm

    tl.store(
        S + offs_dm[:, None] * stride_sm + offs_dn[None, :] * stride_sd,
        acc,
        mask=(offs_dm[:, None] < D) & (offs_dn[None, :] < D),
    )


@triton.jit
def z_kernel(
    Z, PhiK,
    M, D,
    stride_km, stride_kd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute a BLOCK_D slice of Z = sum_m phi(K[m, :])."""
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


@triton.jit
def out_gemm_kernel(
    O, PhiQ, S, Z,
    M, D, eps: tl.constexpr,
    stride_qm, stride_qd,
    stride_sm, stride_sd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one output tile using blocked phi(Q) @ S.

    The denominator phi(Q) @ Z is accumulated over the same BLOCK_K tiles.
    It is shared by all columns in the CTA's output tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = PhiQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    s_ptrs = S + offs_k[:, None] * stride_sm + offs_n[None, :] * stride_sd

    numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        valid_k = k_start + offs_k < D
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        s = tl.load(
            s_ptrs,
            mask=valid_k[:, None] & (offs_n[None, :] < D),
            other=0.0,
        )
        z = tl.load(Z + k_start + offs_k, mask=valid_k, other=0.0)

        numer = tl.dot(q, s, numer, input_precision="tf32")
        denom += tl.sum(q * z[None, :], axis=1)

        q_ptrs += BLOCK_K * stride_qd
        s_ptrs += BLOCK_K * stride_sm

    out = numer / (denom[:, None] + eps)
    tl.store(
        O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_od,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < D),
    )


_kv_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=nw,
            num_stages=1,
        )
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
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
            num_stages=1,
        )
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
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

    if autotune:
        kv_grid = lambda meta: (
            triton.cdiv(D, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
        )
        _kv_kernel_autotuned[kv_grid](
            S, PhiK, V, M, D,
            PhiK.stride(0), PhiK.stride(1),
            V.stride(0), V.stride(1),
            S.stride(0), S.stride(1),
        )
        _launch_z(Z, PhiK, M, D)

        out_grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
        )
        _out_kernel_autotuned[out_grid](
            O, PhiQ, S, Z, M, D, float(eps),
            PhiQ.stride(0), PhiQ.stride(1),
            S.stride(0), S.stride(1),
            O.stride(0), O.stride(1),
        )
    else:
        kv_cfg = _DEFAULT_KV_CONFIG
        # `block_size` is a generic framework argument (1024 when absent from
        # the case).  It is not an algorithm parameter for this operator and
        # must not override the GEMM tile.
        out_cfg = dict(_DEFAULT_OUT_CONFIG)

        kv_grid = (
            triton.cdiv(D, kv_cfg["BLOCK_M"]),
            triton.cdiv(D, kv_cfg["BLOCK_N"]),
        )
        kv_gemm_kernel[kv_grid](
            S, PhiK, V, M, D,
            PhiK.stride(0), PhiK.stride(1),
            V.stride(0), V.stride(1),
            S.stride(0), S.stride(1),
            BLOCK_M=kv_cfg["BLOCK_M"],
            BLOCK_N=kv_cfg["BLOCK_N"],
            BLOCK_K=kv_cfg["BLOCK_K"],
            num_warps=kv_cfg["num_warps"],
            num_stages=kv_cfg["num_stages"],
        )
        _launch_z(Z, PhiK, M, D)

        out_grid = (
            triton.cdiv(M, out_cfg["BLOCK_M"]),
            triton.cdiv(D, out_cfg["BLOCK_N"]),
        )
        out_gemm_kernel[out_grid](
            O, PhiQ, S, Z, M, D, float(eps),
            PhiQ.stride(0), PhiQ.stride(1),
            S.stride(0), S.stride(1),
            O.stride(0), O.stride(1),
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
