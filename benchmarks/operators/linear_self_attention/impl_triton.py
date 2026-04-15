import torch
import triton
import triton.language as tl

_LAST_CONFIG = None


@triton.jit
def _phi(x):
    return tl.where(x > 0, x + 1.0, tl.exp(x))


@triton.jit
def linear_attention_kv_kernel(
    S_ptr,
    K_ptr,
    V_ptr,
    M,
    D,
    stride_km,
    stride_kd,
    stride_vm,
    stride_vd,
    stride_sm,
    stride_sd,
    BLOCK_M: tl.constexpr,
):
    pid_d0 = tl.program_id(0)
    pid_d1 = tl.program_id(1)

    acc = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        k = tl.load(
            K_ptr + offs_m * stride_km + pid_d0 * stride_kd,
            mask=mask_m,
            other=-float("inf"),
        )
        v = tl.load(
            V_ptr + offs_m * stride_vm + pid_d1 * stride_vd,
            mask=mask_m,
            other=0.0,
        )

        phi_k = _phi(k)
        acc += tl.sum(phi_k * v, axis=0)

    tl.store(S_ptr + pid_d0 * stride_sm + pid_d1 * stride_sd, acc)


@triton.jit
def linear_attention_z_kernel(
    Z_ptr,
    K_ptr,
    M,
    D,
    stride_km,
    stride_kd,
    stride_zd,
    BLOCK_M: tl.constexpr,
):
    pid_d = tl.program_id(0)

    acc = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        k = tl.load(
            K_ptr + offs_m * stride_km + pid_d * stride_kd,
            mask=mask_m,
            other=-float("inf"),
        )

        phi_k = _phi(k)
        acc += tl.sum(phi_k, axis=0)

    tl.store(Z_ptr + pid_d * stride_zd, acc)


@triton.jit
def linear_attention_out_kernel(
    O_ptr,
    Q_ptr,
    S_ptr,
    Z_ptr,
    M,
    D,
    eps,
    stride_qm,
    stride_qd,
    stride_om,
    stride_od,
    stride_sm,
    stride_sd,
    stride_zd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_do = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_do = pid_do * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_do = offs_do < D

    numer = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for d_idx in tl.range(0, D, 1):
        q = tl.load(
            Q_ptr + offs_m * stride_qm + d_idx * stride_qd,
            mask=mask_m,
            other=-float("inf"),
        )
        s = tl.load(
            S_ptr + d_idx * stride_sm + offs_do * stride_sd,
            mask=mask_do,
            other=0.0,
        )
        z = tl.load(Z_ptr + d_idx * stride_zd)

        phi_q = _phi(q)

        numer += phi_q[:, None] * s[None, :]
        denom += phi_q * z

    out = numer / (denom[:, None] + eps)

    out_mask = mask_m[:, None] & mask_do[None, :]
    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_do[None, :] * stride_od,
        out,
        mask=out_mask,
    )


def run(
    Q,
    K,
    V,
    eps: float = 1e-6,
    BLOCK_M: int = 32,
    BLOCK_D: int = 16,
    block_size: int = None,
    autotune: bool = False,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_M = int(block_size)

    BLOCK_M = int(BLOCK_M)
    BLOCK_D = int(BLOCK_D)

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.ndim == 2 and K.ndim == 2 and V.ndim == 2
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    # The scalar-reduction kernels do not need Triton autotune here.
    # The launch parameters below are fixed and deterministic.
    linear_attention_kv_kernel[(D, D)](
        S,
        K,
        V,
        M,
        D,
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        S.stride(0),
        S.stride(1),
        BLOCK_M=BLOCK_M,
        num_warps=1,
        num_stages=1,
    )

    linear_attention_z_kernel[(D,)](
        Z,
        K,
        M,
        D,
        K.stride(0),
        K.stride(1),
        Z.stride(0),
        BLOCK_M=BLOCK_M,
        num_warps=1,
        num_stages=1,
    )

    linear_attention_out_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))](
        O,
        Q,
        S,
        Z,
        M,
        D,
        float(eps),
        Q.stride(0),
        Q.stride(1),
        O.stride(0),
        O.stride(1),
        S.stride(0),
        S.stride(1),
        Z.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=1,
        num_stages=1,
    )

    _LAST_CONFIG = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_D": BLOCK_D,
        "num_warps": 1,
        "num_stages": 1,
        "eps": float(eps),
        "kernel_style": "scalar_reduction",
    }
    return O


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    eps = 1e-6
    BLOCK_M = 32
    BLOCK_D = 16

    assert Q.is_cuda and K.is_cuda and V.is_cuda and output.is_cuda
    assert Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32
    assert output.dtype == torch.float32
    assert Q.shape == (M, d)
    assert K.shape == (M, d)
    assert V.shape == (M, d)
    assert output.shape == (M, d)

    S = torch.empty((d, d), device=Q.device, dtype=torch.float32)
    Z = torch.empty((d,), device=Q.device, dtype=torch.float32)

    linear_attention_kv_kernel[(d, d)](
        S,
        K,
        V,
        M,
        d,
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        S.stride(0),
        S.stride(1),
        BLOCK_M=BLOCK_M,
        num_warps=1,
        num_stages=1,
    )

    linear_attention_z_kernel[(d,)](
        Z,
        K,
        M,
        d,
        K.stride(0),
        K.stride(1),
        Z.stride(0),
        BLOCK_M=BLOCK_M,
        num_warps=1,
        num_stages=1,
    )

    linear_attention_out_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(d, BLOCK_D))](
        output,
        Q,
        S,
        Z,
        M,
        d,
        float(eps),
        Q.stride(0),
        Q.stride(1),
        output.stride(0),
        output.stride(1),
        S.stride(0),
        S.stride(1),
        Z.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=1,
        num_stages=1,
    )


def get_last_config() -> dict | None:
    return _LAST_CONFIG