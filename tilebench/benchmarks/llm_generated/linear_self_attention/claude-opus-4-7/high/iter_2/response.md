Looking at the trajectory:
- Triton iter 0 was verify-clean at 1.5%; iter 1 broke verify (SPLIT_K + atomic_add of fp32 caused non-deterministic precision issues with the strict 1e-4 rtol).
- cuTile iter 1 was verify-clean at 0.1% — extremely slow due to SPLIT_K with many atomics.

**Strategy:** Revert to iter 0's pattern (single-sweep, no SPLIT_K, store not atomic_add) and increase `BLOCK_M` to 128 for better arithmetic intensity. Keep `input_precision="ieee"` for fp32 to stay within tolerance.

```python title="impl_triton.py"
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

    BLOCK_M_KV  = 128
    BLOCK_D_KV  = 64
    BLOCK_M_OUT = 128
    BLOCK_D_OUT = 64
    BLOCK_K_OUT = 32
    num_warps   = 4
    num_stages  = 3

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
               NUM_M: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)

    acc   = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_D,),         0.0, dtype=np.float32)

    offs_i = pid_i * BLOCK_D + ct.arange(BLOCK_D, dtype=np.int32)
    mask_i = offs_i < D

    for mb in range(NUM_M):
        offs_m = mb * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
        mask_m = offs_m < M

        k = ct.load(K, index=(mb, pid_i), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.NEG_INF)
        phi_k = ct.where(k > 0.0, k + 1.0, ct.exp(k))
        # mask out OOB rows (NEG_INF gave exp(-inf)=0 already, but be safe on cols)
        phi_k = ct.where(mask_m[:, None], phi_k, 0.0)

        v = ct.load(V, index=(mb, pid_j), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)

        acc = ct.mma(ct.transpose(phi_k), v, acc)

        if pid_j == 0:
            z_acc = z_acc + ct.sum(phi_k, axis=0)

    ct.store(S, index=(pid_i, pid_j), tile=acc)
    if pid_j == 0:
        ct.store(Z, index=(pid_i,), tile=z_acc)


@ct.kernel(occupancy=2)
def _out_kernel(Q, S, Z, O, eps_val, M, D,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt,
                NUM_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    num = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    den = ct.full((BLOCK_M,),         0.0, dtype=np.float32)

    for kb in range(NUM_K):
        q = ct.load(Q, index=(pid_m, kb), shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.NEG_INF)
        phi_q = ct.where(q > 0.0, q + 1.0, ct.exp(q))

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

    S = torch.empty((D, D), dtype=torch.float32, device=Q.device)
    Z = torch.empty((D,),   dtype=torch.float32, device=Q.device)
    O = torch.empty((M, D), dtype=torch.float32, device=Q.device)

    BLOCK_M_KV  = 128
    BLOCK_D_KV  = 64
    BLOCK_M_OUT = 128
    BLOCK_N_OUT = 64
    BLOCK_K_OUT = 32

    NUM_M = (M + BLOCK_M_KV  - 1) // BLOCK_M_KV
    NUM_K = (D + BLOCK_K_OUT - 1) // BLOCK_K_OUT

    stream = torch.cuda.current_stream()

    grid_kv = (
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        1,
    )
    ct.launch(stream, grid_kv, _kv_kernel,
              (K, V, S, Z, M, D, BLOCK_M_KV, BLOCK_D_KV, NUM_M))

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
        "BLOCK_M_OUT": BLOCK_M_OUT,
        "BLOCK_N_OUT": BLOCK_N_OUT,
        "BLOCK_K_OUT": BLOCK_K_OUT,
        "occupancy":   2,
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted Triton to the verify-clean iter-0 baseline (single-sweep, no atomics, `input_precision="ieee"`) and bumped `BLOCK_M` from 64 → 128 in both kernels with `num_stages=3` for better mainloop pipelining and arithmetic intensity. For cuTile, eliminated the SPLIT_K + atomic_add path that made iter 1 slow and use a direct single-sweep store like the Triton version, with the same larger block sizes.
