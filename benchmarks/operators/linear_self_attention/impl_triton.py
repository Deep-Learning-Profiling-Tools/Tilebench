"""Triton linear self-attention via 3 kernels:

Stage 1 (`_kv_kernel`): Compute S = phi(K)^T @ V into a (D, D) buffer.
  Grid (D, D); each CTA owns one S[d0, d1] scalar and reduces along M.

Stage 2 (`_z_kernel`):  Compute Z = sum_m phi(K[m, :]) into a (D,) buffer.
  Grid (D,); each CTA owns one Z[d] scalar.

Stage 3 (`_out_kernel`): Compute O = (phi(Q) @ S) / (phi(Q) @ Z + eps).
  Grid (cdiv(M, BLOCK_M), cdiv(D, BLOCK_D)); each CTA emits a
  (BLOCK_M, BLOCK_D) output tile via a D-step inner loop accumulating
  outer products of (phi_q, s_row) and a denominator scalar (phi_q @ z).

Only stage 3 is autotuned — it dominates cost for non-tiny D (its inner
loop length scales with D and it walks all of M). Stages 1/2 are simple
reductions over a fixed grid and use a fixed config to mirror the cuTile
side and keep the search space narrow.
"""
import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {
    "KV_BLOCK_M": 32,
    "BLOCK_M":    32,
    "BLOCK_D":    16,
    "num_warps":  4,
    "num_stages": 2,
}


@triton.jit
def _phi(x):
    # phi(x) = ELU(x) + 1
    return tl.where(x > 0, x + 1.0, tl.exp(x))


@triton.jit
def _kv_kernel(
    S_ptr, K_ptr, V_ptr,
    M, D,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_sm, stride_sd,
    BLOCK_M: tl.constexpr,
):
    # Each program computes one scalar S[d0, d1].
    pid_d0 = tl.program_id(0)
    pid_d1 = tl.program_id(1)

    acc = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        k = tl.load(K_ptr + offs_m * stride_km + pid_d0 * stride_kd,
                    mask=mask_m, other=-float("inf"))
        v = tl.load(V_ptr + offs_m * stride_vm + pid_d1 * stride_vd,
                    mask=mask_m, other=0.0)

        phi_k = _phi(k)
        acc += tl.sum(phi_k * v, axis=0)

    tl.store(S_ptr + pid_d0 * stride_sm + pid_d1 * stride_sd, acc)


@triton.jit
def _z_kernel(
    Z_ptr, K_ptr,
    M, D,
    stride_km, stride_kd,
    stride_zd,
    BLOCK_M: tl.constexpr,
):
    # Each program computes one scalar Z[d].
    pid_d = tl.program_id(0)

    acc = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        k = tl.load(K_ptr + offs_m * stride_km + pid_d * stride_kd,
                    mask=mask_m, other=-float("inf"))
        phi_k = _phi(k)
        acc += tl.sum(phi_k, axis=0)

    tl.store(Z_ptr + pid_d * stride_zd, acc)


@triton.jit
def _out_kernel(
    O_ptr, Q_ptr, S_ptr, Z_ptr,
    M, D, eps: tl.constexpr,
    stride_qm, stride_qd,
    stride_om, stride_od,
    stride_sm, stride_sd,
    stride_zd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program computes O tile [BLOCK_M, BLOCK_D].
    pid_m = tl.program_id(0)
    pid_do = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_do = pid_do * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_do = offs_do < D

    numer = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for d_idx in tl.range(0, D, 1):
        q = tl.load(Q_ptr + offs_m * stride_qm + d_idx * stride_qd,
                    mask=mask_m, other=-float("inf"))
        s = tl.load(S_ptr + d_idx * stride_sm + offs_do * stride_sd,
                    mask=mask_do, other=0.0)
        z = tl.load(Z_ptr + d_idx * stride_zd)

        phi_q = _phi(q)

        numer += phi_q[:, None] * s[None, :]
        denom += phi_q * z

    out = numer / (denom[:, None] + eps)

    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_do[None, :] * stride_od,
        out,
        mask=mask_m[:, None] & mask_do[None, :],
    )


_out_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_D": bd},
            num_warps=nw,
            num_stages=ns,
        )
        for bm in [16, 32, 64]
        for bd in [8, 16, 32]
        for nw in [1, 2, 4]
        for ns in [1, 2, 3]
    ],
    key=["M", "D"],
)(_out_kernel)


def _launch_kv_z(Q, K, V, M, D, KV_BLOCK_M):
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    _kv_kernel[(D, D)](
        S, K, V, M, D,
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        S.stride(0), S.stride(1),
        BLOCK_M=KV_BLOCK_M,
        num_warps=1, num_stages=1,
    )
    _z_kernel[(D,)](
        Z, K, M, D,
        K.stride(0), K.stride(1),
        Z.stride(0),
        BLOCK_M=KV_BLOCK_M,
        num_warps=1, num_stages=1,
    )
    return S, Z


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)

    cfg = _DEFAULT_CONFIG
    S, Z = _launch_kv_z(Q, K, V, M, D, cfg["KV_BLOCK_M"])

    if autotune:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        _out_kernel_autotuned[grid](
            O, Q, S, Z, M, D, float(eps),
            Q.stride(0), Q.stride(1),
            O.stride(0), O.stride(1),
            S.stride(0), S.stride(1),
            Z.stride(0),
        )
    else:
        BLOCK_M = int(block_size) if block_size is not None else cfg["BLOCK_M"]
        BLOCK_D = cfg["BLOCK_D"]
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
        _out_kernel[grid](
            O, Q, S, Z, M, D, float(eps),
            Q.stride(0), Q.stride(1),
            O.stride(0), O.stride(1),
            S.stride(0), S.stride(1),
            Z.stride(0),
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return O


def get_last_config() -> dict | None:
    cfg = getattr(_out_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M":     cfg.kwargs["BLOCK_M"],
        "BLOCK_D":     cfg.kwargs["BLOCK_D"],
        "num_warps":   cfg.num_warps,
        "num_stages":  cfg.num_stages,
    }
