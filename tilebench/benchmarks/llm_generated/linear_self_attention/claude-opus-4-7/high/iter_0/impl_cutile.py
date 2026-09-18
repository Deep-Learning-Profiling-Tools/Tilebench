import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel
def _kv_kernel(K, V, S, Z, M, D,
               BLOCK_M: ConstInt, BLOCK_D: ConstInt, NUM_M: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)

    acc_d = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float64)
    z_acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)

    offs_i = pid_i * BLOCK_D + ct.arange(BLOCK_D, dtype=np.int32)
    offs_j = pid_j * BLOCK_D + ct.arange(BLOCK_D, dtype=np.int32)
    mask_i = offs_i < D
    mask_j = offs_j < D

    for mb in range(NUM_M):
        offs_m = mb * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
        mask_m = offs_m < M

        k = ct.load(K, index=(mb, pid_i), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)
        phi_k = ct.where(k > 0.0, k + 1.0, ct.exp(k))
        # phi(loaded-0) = 1, must zero out OOB explicitly
        valid = mask_m[:, None] & mask_i[None, :]
        phi_k = ct.where(valid, phi_k, 0.0)

        v = ct.load(V, index=(mb, pid_j), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)

        # fp64 matmul for IEEE-fp32-equivalent accuracy
        phi_k_d = ct.astype(phi_k, np.float64)
        v_d     = ct.astype(v,     np.float64)
        acc_d = ct.mma(ct.transpose(phi_k_d), v_d, acc_d)

        if pid_j == 0:
            z_acc = z_acc + ct.sum(phi_k, axis=0)

    acc_f = ct.astype(acc_d, np.float32)
    ct.store(S, index=(pid_i, pid_j), tile=acc_f)

    if pid_j == 0:
        ct.store(Z, index=(pid_i,), tile=z_acc)


@ct.kernel
def _out_kernel(Q, S, Z, O, eps_val, M, D,
                BLOCK_M: ConstInt, BLOCK_D: ConstInt, BLOCK_K: ConstInt,
                NUM_K: ConstInt):
    pid_m = ct.bid(0)
    pid_d = ct.bid(1)

    num_d = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=np.float64)
    den_d = ct.full((BLOCK_M,), 0.0, dtype=np.float64)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    mask_m = offs_m < M

    for kb in range(NUM_K):
        offs_k = kb * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)
        mask_k = offs_k < D

        q = ct.load(Q, index=(pid_m, kb), shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO)
        phi_q = ct.where(q > 0.0, q + 1.0, ct.exp(q))
        phi_q = ct.where(mask_m[:, None] & mask_k[None, :], phi_q, 0.0)

        s = ct.load(S, index=(kb, pid_d), shape=(BLOCK_K, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)

        phi_q_d = ct.astype(phi_q, np.float64)
        s_d     = ct.astype(s,     np.float64)
        num_d = ct.mma(phi_q_d, s_d, num_d)

        z = ct.load(Z, index=(kb,), shape=(BLOCK_K,),
                    padding_mode=ct.PaddingMode.ZERO)
        z_d = ct.astype(z, np.float64)
        den_d = den_d + ct.sum(phi_q_d * z_d[None, :], axis=1)

    num = ct.astype(num_d, np.float32)
    den = ct.astype(den_d, np.float32)
    out = num / (den[:, None] + eps_val)
    ct.store(O, index=(pid_m, pid_d), tile=out)


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
    BLOCK_M_OUT = 64
    BLOCK_D_OUT = 64
    BLOCK_K_OUT = 64
    occupancy   = 2

    NUM_M = (M + BLOCK_M_KV - 1) // BLOCK_M_KV
    NUM_K = (D + BLOCK_K_OUT - 1) // BLOCK_K_OUT

    stream = torch.cuda.current_stream()

    grid_kv = (
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        (D + BLOCK_D_KV - 1) // BLOCK_D_KV,
        1,
    )
    kernel_kv = _kv_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid_kv, kernel_kv,
              (K, V, S, Z, M, D, BLOCK_M_KV, BLOCK_D_KV, NUM_M))

    grid_out = (
        (M + BLOCK_M_OUT - 1) // BLOCK_M_OUT,
        (D + BLOCK_D_OUT - 1) // BLOCK_D_OUT,
        1,
    )
    kernel_out = _out_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid_out, kernel_out,
              (Q, S, Z, O, float(eps), M, D,
               BLOCK_M_OUT, BLOCK_D_OUT, BLOCK_K_OUT, NUM_K))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M_KV":  BLOCK_M_KV,
        "BLOCK_D_KV":  BLOCK_D_KV,
        "BLOCK_M_OUT": BLOCK_M_OUT,
        "BLOCK_D_OUT": BLOCK_D_OUT,
        "BLOCK_K_OUT": BLOCK_K_OUT,
        "occupancy":   occupancy,
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
