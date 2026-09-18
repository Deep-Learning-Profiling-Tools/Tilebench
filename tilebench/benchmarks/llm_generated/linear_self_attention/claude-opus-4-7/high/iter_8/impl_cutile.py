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
    BLOCK_M_OUT = 64
    BLOCK_N_OUT = 256
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
