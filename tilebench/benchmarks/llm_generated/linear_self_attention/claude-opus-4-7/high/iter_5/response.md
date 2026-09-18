Reverting both to iter 3 (best verify-clean). For triton, I'll keep `tf32x3` precision (needed for fp32 verify) but bump `BLOCK_N_OUT` to 128 and `num_warps` to 8 to better saturate compute on the dominant output GEMM. For cutile, I'll keep the iter 3 atomic-add approach (only verify-clean approach so far) with the same configuration.

```python title="impl_triton.py"
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

        acc = tl.dot(tl.trans(phi_k), v, acc, input_precision="tf32x3")

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
        num = tl.dot(phi_q, s, num, input_precision="tf32x3")

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
    BLOCK_M_OUT = 128
    BLOCK_N_OUT = 128
    BLOCK_K_OUT = 32
    num_warps_kv   = 4
    num_stages_kv  = 2
    num_warps_out  = 8
    num_stages_out = 2

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
        "precision":   "tf32x3",
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _kv_kernel(K, V, S, Z, M, D,
               BLOCK_M: ConstInt, BLOCK_D: ConstInt,
               TILES_PER_SPLIT: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)
    pid_k = ct.bid(2)

    acc   = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_D,),         0.0, dtype=np.float32)

    offs_i = pid_i * BLOCK_D + ct.arange(BLOCK_D, dtype=np.int32)
    offs_j = pid_j * BLOCK_D + ct.arange(BLOCK_D, dtype=np.int32)
    mask_i = offs_i < D
    mask_j = offs_j < D

    for mb_local in range(TILES_PER_SPLIT):
        mb = pid_k * TILES_PER_SPLIT + mb_local

        offs_m = mb * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
        mask_m = offs_m < M
        valid_mi = mask_m[:, None] & mask_i[None, :]

        k = ct.load(K, index=(mb, pid_i), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)
        phi_k = ct.where(k > 0.0, k + 1.0, ct.exp(k))
        phi_k = ct.where(valid_mi, phi_k, 0.0)

        v = ct.load(V, index=(mb, pid_j), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)

        acc = ct.mma(ct.transpose(phi_k), v, acc)

        if pid_j == 0:
            z_acc = z_acc + ct.sum(phi_k, axis=0)

    valid_ij = mask_i[:, None] & mask_j[None, :]
    acc = ct.where(valid_ij, acc, 0.0)
    ct.atomic_add(S, (offs_i[:, None], offs_j[None, :]), acc)

    if pid_j == 0:
        z_acc = ct.where(mask_i, z_acc, 0.0)
        ct.atomic_add(Z, (offs_i,), z_acc)


@ct.kernel(occupancy=2)
def _out_kernel(Q, S, Z, O, eps_val, M, D,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt,
                NUM_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    num = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    den = ct.full((BLOCK_M,),         0.0, dtype=np.float32)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    mask_m = offs_m < M

    for kb in range(NUM_K):
        offs_k = kb * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)
        mask_k = offs_k < D
        valid_qk = mask_m[:, None] & mask_k[None, :]

        q = ct.load(Q, index=(pid_m, kb), shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO)
        phi_q = ct.where(q > 0.0, q + 1.0, ct.exp(q))
        phi_q = ct.where(valid_qk, phi_q, 0.0)

        s = ct.load(S, index=(kb, pid_n), shape=(BLOCK_K, BLOCK_N),
                    padding_mode=ct.PaddingMode.ZERO)
        num = ct.mma(phi_q, s, num)

        z = ct.load(Z, index=(kb,), shape=(BLOCK_K,),
                    padding_mode=ct.PaddingMode.ZERO)
        den = den + ct.sum(phi_q * z[None, :], axis=1)

    out = num / (den[:, None] + eps_val)
    ct.store(O, index=(pid_m, pid_n), tile=out)


def run(Q, K, V, eps=1e-6, **kwargs):
    M, D = Q.shape
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    S = torch.zeros((D, D), dtype=torch.float32, device=Q.device)
    Z = torch.zeros((D,),   dtype=torch.float32, device=Q.device)
    O = torch.empty((M, D), dtype=torch.float32, device=Q.device)

    BLOCK_M_KV  = 128
    BLOCK_D_KV  = 64
    SPLIT_K     = 8
    BLOCK_M_OUT = 128
    BLOCK_N_OUT = 128
    BLOCK_K_OUT = 32

    NUM_M_TILES     = (M + BLOCK_M_KV - 1) // BLOCK_M_KV
    TILES_PER_SPLIT = (NUM_M_TILES + SPLIT_K - 1) // SPLIT_K
    NUM_K           = (D + BLOCK_K_OUT - 1) // BLOCK_K_OUT

    stream = torch.cuda.current_stream()

    grid_kv = (
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        SPLIT_K,
    )
    ct.launch(stream, grid_kv, _kv_kernel,
              (K, V, S, Z, M, D, BLOCK_M_KV, BLOCK_D_KV, TILES_PER_SPLIT))

    grid_out = (
        (M + BLOCK_M_OUT - 1) // BLOCK_M_OUT,
        (D + BLOCK_N_OUT - 1) // BLOCK_N_OUT,
        1,
    )
    ct.launch(stream, grid_out, _out_kernel,
              (Q, S, Z, O, float(eps), M, D,
               BLOCK_M_OUT, BLOCK_N_OUT, BLOCK_K_OUT, NUM_K))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M_KV":  BLOCK_M_KV,
        "BLOCK_D_KV":  BLOCK_D_KV,
        "SPLIT_K":     SPLIT_K,
        "BLOCK_M_OUT": BLOCK_M_OUT,
        "BLOCK_N_OUT": BLOCK_N_OUT,
        "BLOCK_K_OUT": BLOCK_K_OUT,
        "occupancy":   2,
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
