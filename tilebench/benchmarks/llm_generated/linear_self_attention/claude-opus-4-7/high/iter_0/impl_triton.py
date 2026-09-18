import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _kv_kernel(K_ptr, V_ptr, S_ptr, Z_ptr, M, D,
               BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_j = pid_j * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)

    mask_i = offs_i < D
    mask_j = offs_j < D

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_m in range(0, M, BLOCK_M):
        m = start_m + offs_m
        mask_m = m < M

        k_ptrs = K_ptr + m[:, None] * D + offs_i[None, :]
        # OOB load gets -inf → phi(-inf)=exp(-inf)=0 → no contribution
        k = tl.load(k_ptrs, mask=mask_m[:, None] & mask_i[None, :],
                    other=float('-inf'))
        phi_k = tl.where(k > 0.0, k + 1.0, tl.exp(k))

        v_ptrs = V_ptr + m[:, None] * D + offs_j[None, :]
        v = tl.load(v_ptrs, mask=mask_m[:, None] & mask_j[None, :], other=0.0)

        acc = tl.dot(tl.trans(phi_k), v, acc, input_precision="ieee")

        if pid_j == 0:
            z_acc += tl.sum(phi_k, axis=0)

    s_ptrs = S_ptr + offs_i[:, None] * D + offs_j[None, :]
    tl.store(s_ptrs, acc, mask=mask_i[:, None] & mask_j[None, :])

    if pid_j == 0:
        tl.store(Z_ptr + offs_i, z_acc, mask=mask_i)


@triton.jit
def _out_kernel(Q_ptr, S_ptr, Z_ptr, O_ptr, eps, M, D,
                BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_d = offs_d < D

    num = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    den = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_k in range(0, D, BLOCK_K):
        k = start_k + offs_k_base
        mask_k = k < D

        q_ptrs = Q_ptr + offs_m[:, None] * D + k[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :],
                    other=float('-inf'))
        phi_q = tl.where(q > 0.0, q + 1.0, tl.exp(q))

        s_ptrs = S_ptr + k[:, None] * D + offs_d[None, :]
        s = tl.load(s_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
        num = tl.dot(phi_q, s, num, input_precision="ieee")

        z = tl.load(Z_ptr + k, mask=mask_k, other=0.0)
        den += tl.sum(phi_q * z[None, :], axis=1)

    out = num / (den[:, None] + eps)
    o_ptrs = O_ptr + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


def run(Q, K, V, eps=1e-6, **kwargs):
    M, D = Q.shape
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    S = torch.empty((D, D), dtype=torch.float32, device=Q.device)
    Z = torch.empty((D,), dtype=torch.float32, device=Q.device)
    O = torch.empty((M, D), dtype=torch.float32, device=Q.device)

    BLOCK_M_KV  = 64
    BLOCK_D_KV  = 64
    BLOCK_M_OUT = 64
    BLOCK_D_OUT = 64
    BLOCK_K_OUT = 64
    num_warps   = 4
    num_stages  = 2

    grid_kv = (triton.cdiv(D, BLOCK_D_KV), triton.cdiv(D, BLOCK_D_KV))
    _kv_kernel[grid_kv](
        K, V, S, Z, M, D,
        BLOCK_M=BLOCK_M_KV, BLOCK_D=BLOCK_D_KV,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid_out = (triton.cdiv(M, BLOCK_M_OUT), triton.cdiv(D, BLOCK_D_OUT))
    _out_kernel[grid_out](
        Q, S, Z, O, float(eps), M, D,
        BLOCK_M=BLOCK_M_OUT, BLOCK_D=BLOCK_D_OUT, BLOCK_K=BLOCK_K_OUT,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M_KV":  BLOCK_M_KV,
        "BLOCK_D_KV":  BLOCK_D_KV,
        "BLOCK_M_OUT": BLOCK_M_OUT,
        "BLOCK_D_OUT": BLOCK_D_OUT,
        "BLOCK_K_OUT": BLOCK_K_OUT,
        "num_warps":   num_warps,
        "num_stages":  num_stages,
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
