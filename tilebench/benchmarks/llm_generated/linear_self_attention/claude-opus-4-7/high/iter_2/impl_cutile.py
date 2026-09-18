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
