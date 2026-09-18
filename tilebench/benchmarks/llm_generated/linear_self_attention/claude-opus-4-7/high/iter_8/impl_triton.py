import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _kv_kernel(K_ptr, V_ptr, S_ptr, Z_ptr, M, D,
               BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
               SPLIT_K: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_i = pid_i * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_j = pid_j * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_i = offs_i < D
    mask_j = offs_j < D

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    m_per_split = tl.cdiv(M, SPLIT_K)
    m_start = pid_k * m_per_split
    m_end = tl.minimum(m_start + m_per_split, M)

    for start_m in range(m_start, m_end, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_end

        k = tl.load(K_ptr + offs_m[:, None] * D + offs_i[None, :],
                    mask=mask_m[:, None] & mask_i[None, :],
                    other=float('-inf'))
        phi_k = tl.where(k > 0.0, k + 1.0, tl.exp(k))

        v = tl.load(V_ptr + offs_m[:, None] * D + offs_j[None, :],
                    mask=mask_m[:, None] & mask_j[None, :],
                    other=0.0)

        acc = tl.dot(tl.trans(phi_k), v, acc, input_precision="tf32")

        if pid_j == 0:
            z_acc += tl.sum(phi_k, axis=0)

    s_ptrs = S_ptr + offs_i[:, None] * D + offs_j[None, :]
    s_mask = mask_i[:, None] & mask_j[None, :]
    if SPLIT_K == 1:
        tl.store(s_ptrs, acc, mask=s_mask)
        if pid_j == 0:
            tl.store(Z_ptr + offs_i, z_acc, mask=mask_i)
    else:
        tl.atomic_add(s_ptrs, acc, mask=s_mask)
        if pid_j == 0:
            tl.atomic_add(Z_ptr + offs_i, z_acc, mask=mask_i)


@triton.jit
def _out_kernel(Q_ptr, S_ptr, Z_ptr, O_ptr, eps, M, D,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < D

    num = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    den = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_k in range(0, D, BLOCK_K):
        k = start_k + offs_k
        mask_k = k < D

        q = tl.load(Q_ptr + offs_m[:, None] * D + k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=float('-inf'))
        phi_q = tl.where(q > 0.0, q + 1.0, tl.exp(q))

        s = tl.load(S_ptr + k[:, None] * D + offs_n[None, :],
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)
        num = tl.dot(phi_q, s, num, input_precision="tf32")

        z = tl.load(Z_ptr + k, mask=mask_k, other=0.0)
        den += tl.sum(phi_q * z[None, :], axis=1)

    out = num / (den[:, None] + eps)
    tl.store(O_ptr + offs_m[:, None] * D + offs_n[None, :], out,
             mask=mask_m[:, None] & mask_n[None, :])


def run(Q, K, V, eps=1e-6, **kwargs):
    M, D = Q.shape
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    BLOCK_M_KV  = 128
    BLOCK_D_KV  = 64
    SPLIT_K     = 8
    BLOCK_M_OUT = 64
    BLOCK_N_OUT = 256
    BLOCK_K_OUT = 32
    num_warps_kv   = 4
    num_stages_kv  = 3
    num_warps_out  = 8
    num_stages_out = 3

    if SPLIT_K > 1:
        S = torch.zeros((D, D), dtype=torch.float32, device=Q.device)
        Z = torch.zeros((D,), dtype=torch.float32, device=Q.device)
    else:
        S = torch.empty((D, D), dtype=torch.float32, device=Q.device)
        Z = torch.empty((D,), dtype=torch.float32, device=Q.device)
    O = torch.empty((M, D), dtype=torch.float32, device=Q.device)

    grid_kv = (triton.cdiv(D, BLOCK_D_KV), triton.cdiv(D, BLOCK_D_KV), SPLIT_K)
    _kv_kernel[grid_kv](
        K, V, S, Z, M, D,
        BLOCK_M=BLOCK_M_KV, BLOCK_D=BLOCK_D_KV, SPLIT_K=SPLIT_K,
        num_warps=num_warps_kv, num_stages=num_stages_kv,
    )

    grid_out = (triton.cdiv(M, BLOCK_M_OUT), triton.cdiv(D, BLOCK_N_OUT))
    _out_kernel[grid_out](
        Q, S, Z, O, float(eps), M, D,
        BLOCK_M=BLOCK_M_OUT, BLOCK_N=BLOCK_N_OUT, BLOCK_K=BLOCK_K_OUT,
        num_warps=num_warps_out, num_stages=num_stages_out,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M_KV":  BLOCK_M_KV,
        "BLOCK_D_KV":  BLOCK_D_KV,
        "SPLIT_K":     SPLIT_K,
        "BLOCK_M_OUT": BLOCK_M_OUT,
        "BLOCK_N_OUT": BLOCK_N_OUT,
        "BLOCK_K_OUT": BLOCK_K_OUT,
        "num_warps_kv":   num_warps_kv,
        "num_stages_kv":  num_stages_kv,
        "num_warps_out":  num_warps_out,
        "num_stages_out": num_stages_out,
        "precision":   "tf32",
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
