import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _kv_kernel(K, V, S_part, Z_part, M, D,
               BLOCK_M: ConstInt, BLOCK_D: ConstInt,
               TILES_PER_SPLIT: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)
    pid_k = ct.bid(2)

    acc   = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_D,),         0.0, dtype=np.float32)

    for mb_local in range(TILES_PER_SPLIT):
        mb = pid_k * TILES_PER_SPLIT + mb_local

        # NEG_INF padding => phi(-inf) = exp(-inf) = 0, OOB auto-zeroed.
        k = ct.load(K, index=(mb, pid_i), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.NEG_INF)
        phi_k = ct.where(k > 0.0, k + 1.0, ct.exp(k))

        v = ct.load(V, index=(mb, pid_j), shape=(BLOCK_M, BLOCK_D),
                    padding_mode=ct.PaddingMode.ZERO)

        acc = ct.mma(ct.transpose(phi_k), v, acc)

        if pid_j == 0:
            z_acc = z_acc + ct.sum(phi_k, axis=0)

    # Static-indexed store into partial buffer — no atomics.
    acc_3d = acc.reshape((1, BLOCK_D, BLOCK_D))
    ct.store(S_part, index=(pid_k, pid_i, pid_j), tile=acc_3d)

    if pid_j == 0:
        z_2d = z_acc.reshape((1, BLOCK_D))
        ct.store(Z_part, index=(pid_k, pid_i), tile=z_2d)


@ct.kernel(occupancy=2)
def _reduce_kernel(S_part, Z_part, S, Z, D,
                   SPLIT_K: ConstInt, BLOCK_D: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)

    acc = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    for k_idx in range(SPLIT_K):
        s_part = ct.load(S_part, index=(k_idx, pid_i, pid_j),
                         shape=(1, BLOCK_D, BLOCK_D),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = acc + s_part.reshape((BLOCK_D, BLOCK_D))

    ct.store(S, index=(pid_i, pid_j), tile=acc)

    if pid_j == 0:
        z_acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
        for k_idx in range(SPLIT_K):
            z_part = ct.load(Z_part, index=(k_idx, pid_i),
                             shape=(1, BLOCK_D),
                             padding_mode=ct.PaddingMode.ZERO)
            z_acc = z_acc + z_part.reshape((BLOCK_D,))
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

    BLOCK_M_KV  = 128
    BLOCK_D_KV  = 64
    SPLIT_K     = 16
    BLOCK_M_OUT = 64
    BLOCK_N_OUT = 256
    BLOCK_K_OUT = 32

    NUM_M_TILES     = (M + BLOCK_M_KV - 1) // BLOCK_M_KV
    TILES_PER_SPLIT = (NUM_M_TILES + SPLIT_K - 1) // SPLIT_K
    NUM_K           = (D + BLOCK_K_OUT - 1) // BLOCK_K_OUT
    NUM_I_TILES     = (D + BLOCK_D_KV - 1) // BLOCK_D_KV
    D_PAD           = NUM_I_TILES * BLOCK_D_KV

    S_part = torch.empty((SPLIT_K, D_PAD, D_PAD), dtype=torch.float32, device=Q.device)
    Z_part = torch.empty((SPLIT_K, D_PAD),        dtype=torch.float32, device=Q.device)
    S      = torch.empty((D, D),                  dtype=torch.float32, device=Q.device)
    Z      = torch.empty((D,),                    dtype=torch.float32, device=Q.device)
    O      = torch.empty((M, D),                  dtype=torch.float32, device=Q.device)

    stream = torch.cuda.current_stream()

    grid_kv = (NUM_I_TILES, NUM_I_TILES, SPLIT_K)
    ct.launch(stream, grid_kv, _kv_kernel,
              (K, V, S_part, Z_part, M, D,
               BLOCK_M_KV, BLOCK_D_KV, TILES_PER_SPLIT))

    grid_red = (NUM_I_TILES, NUM_I_TILES, 1)
    ct.launch(stream, grid_red, _reduce_kernel,
              (S_part, Z_part, S, Z, D, SPLIT_K, BLOCK_D_KV))

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
        "scheme":      "partial_buf_reduce",
    })
    return O


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
