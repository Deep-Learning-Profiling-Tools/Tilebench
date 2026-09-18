import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _mma_tf32x3(a, b, acc):
    a0 = ct.astype(a, ct.tfloat32)
    b0 = ct.astype(b, ct.tfloat32)

    a0_f = ct.astype(a0, np.float32)
    b0_f = ct.astype(b0, np.float32)

    a1 = ct.astype(a - a0_f, ct.tfloat32)
    b1 = ct.astype(b - b0_f, ct.tfloat32)

    acc = ct.mma(a0, b0, acc)
    acc = ct.mma(a0, b1, acc)
    acc = ct.mma(a1, b0, acc)
    return acc


@ct.kernel
def _phi2_kernel(Q, K, PhiQ, PhiK,
                 N: ConstInt,
                 BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)

    q = ct.astype(
        ct.load(Q, index=(bid,), shape=(BLOCK_SIZE,),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32,
    )
    k = ct.astype(
        ct.load(K, index=(bid,), shape=(BLOCK_SIZE,),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32,
    )

    phi_q = ct.where(q > 0.0, q + 1.0, ct.exp(q))
    phi_k = ct.where(k > 0.0, k + 1.0, ct.exp(k))

    ct.store(PhiQ, index=(bid,), tile=phi_q)
    ct.store(PhiK, index=(bid,), tile=phi_k)


@ct.kernel
def _partial_kv_kernel(PhiK, V, partial_S, partial_Z,
                       M: ConstInt, D: ConstInt,
                       CHUNK_TILES: ConstInt,
                       BLOCK_DI: ConstInt,
                       BLOCK_DJ: ConstInt,
                       KV_BLOCK_M: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)
    pid_s = ct.bid(2)

    acc = ct.full((BLOCK_DI, BLOCK_DJ), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_DI,), 0.0, dtype=np.float32)

    for t in range(0, CHUNK_TILES):
        row_tile = pid_s * CHUNK_TILES + t

        k_phi = ct.astype(
            ct.load(
                PhiK,
                index=(row_tile, pid_i),
                shape=(KV_BLOCK_M, BLOCK_DI),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )
        v_tile = ct.astype(
            ct.load(
                V,
                index=(row_tile, pid_j),
                shape=(KV_BLOCK_M, BLOCK_DJ),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        acc = _mma_tf32x3(ct.transpose(k_phi), v_tile, acc)
        z_acc = z_acc + ct.sum(k_phi, axis=0)

    ct.store(
        partial_S,
        index=(pid_s, pid_i, pid_j),
        tile=ct.reshape(acc, (1, BLOCK_DI, BLOCK_DJ)),
    )

    if pid_j == 0:
        ct.store(
            partial_Z,
            index=(pid_s, pid_i),
            tile=ct.reshape(z_acc, (1, BLOCK_DI)),
        )


@ct.kernel
def _reduce_kv_kernel(partial_S, partial_Z, S, Z,
                      NUM_SPLITS: ConstInt,
                      BLOCK_DI: ConstInt,
                      BLOCK_DJ: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)

    acc = ct.full((BLOCK_DI, BLOCK_DJ), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_DI,), 0.0, dtype=np.float32)

    for s in range(0, NUM_SPLITS):
        ps = ct.load(
            partial_S,
            index=(s, pid_i, pid_j),
            shape=(1, BLOCK_DI, BLOCK_DJ),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.reshape(ct.astype(ps, np.float32), (BLOCK_DI, BLOCK_DJ))

        pz = ct.load(
            partial_Z,
            index=(s, pid_i),
            shape=(1, BLOCK_DI),
            padding_mode=ct.PaddingMode.ZERO,
        )
        z_acc = z_acc + ct.reshape(ct.astype(pz, np.float32), (BLOCK_DI,))

    ct.store(S, index=(pid_i, pid_j), tile=acc)

    if pid_j == 0:
        ct.store(Z, index=(pid_i,), tile=z_acc)


@ct.kernel
def _output_kernel(PhiQ, S, Z, output, eps,
                   M: ConstInt, D: ConstInt,
                   NUM_K_TILES: ConstInt,
                   OUT_BLOCK_M: ConstInt,
                   OUT_BLOCK_D: ConstInt,
                   BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((OUT_BLOCK_M, OUT_BLOCK_D), 0.0, dtype=np.float32)
    denom = ct.full((OUT_BLOCK_M,), 0.0, dtype=np.float32)

    for kb in range(0, NUM_K_TILES):
        q_phi = ct.astype(
            ct.load(
                PhiQ,
                index=(pid_m, kb),
                shape=(OUT_BLOCK_M, BLOCK_K),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        s_tile = ct.astype(
            ct.load(
                S,
                index=(kb, pid_n),
                shape=(BLOCK_K, OUT_BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        acc = _mma_tf32x3(q_phi, s_tile, acc)

        z_tile = ct.astype(
            ct.load(
                Z,
                index=(kb,),
                shape=(BLOCK_K,),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )
        denom = denom + ct.sum(q_phi * z_tile[None, :], axis=1)

    out = acc / (denom[:, None] + eps)
    ct.store(output, index=(pid_m, pid_n), tile=out)


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M = Q.shape[0]
    D = Q.shape[1]
    N = M * D

    output = torch.empty_like(Q)

    PHI_BLOCK_SIZE = 1024
    NUM_SPLITS = 16
    BLOCK_DI = 64
    BLOCK_DJ = 128
    KV_BLOCK_M = 64
    OUT_BLOCK_M = 64
    OUT_BLOCK_D = 128
    BLOCK_K = 64

    PHI_OCCUPANCY = 8
    PARTIAL_OCCUPANCY = 2
    REDUCE_OCCUPANCY = 2
    OUT_OCCUPANCY = 2

    row_tiles = (M + KV_BLOCK_M - 1) // KV_BLOCK_M
    chunk_tiles = (row_tiles + NUM_SPLITS - 1) // NUM_SPLITS
    d_tiles_i = (D + BLOCK_DI - 1) // BLOCK_DI
    d_tiles_j = (D + BLOCK_DJ - 1) // BLOCK_DJ
    out_m_tiles = (M + OUT_BLOCK_M - 1) // OUT_BLOCK_M
    out_n_tiles = (D + OUT_BLOCK_D - 1) // OUT_BLOCK_D
    num_k_tiles = (D + BLOCK_K - 1) // BLOCK_K

    PhiQ = torch.empty_like(Q)
    PhiK = torch.empty_like(K)
    partial_S = torch.empty((NUM_SPLITS, D, D), device=Q.device, dtype=torch.float32)
    partial_Z = torch.empty((NUM_SPLITS, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        ((N + PHI_BLOCK_SIZE - 1) // PHI_BLOCK_SIZE, 1, 1),
        _phi2_kernel.with_hints(occupancy=PHI_OCCUPANCY),
        (Q.reshape(-1), K.reshape(-1), PhiQ.reshape(-1), PhiK.reshape(-1),
         N, PHI_BLOCK_SIZE),
    )

    ct.launch(
        stream,
        (d_tiles_i, d_tiles_j, NUM_SPLITS),
        _partial_kv_kernel.with_hints(occupancy=PARTIAL_OCCUPANCY),
        (PhiK, V, partial_S, partial_Z, M, D,
         chunk_tiles, BLOCK_DI, BLOCK_DJ, KV_BLOCK_M),
    )

    ct.launch(
        stream,
        (d_tiles_i, d_tiles_j, 1),
        _reduce_kv_kernel.with_hints(occupancy=REDUCE_OCCUPANCY),
        (partial_S, partial_Z, S, Z, NUM_SPLITS, BLOCK_DI, BLOCK_DJ),
    )

    ct.launch(
        stream,
        (out_m_tiles, out_n_tiles, 1),
        _output_kernel.with_hints(occupancy=OUT_OCCUPANCY),
        (PhiQ, S, Z, output, float(eps), M, D, num_k_tiles,
         OUT_BLOCK_M, OUT_BLOCK_D, BLOCK_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "PHI_BLOCK_SIZE": PHI_BLOCK_SIZE,
        "NUM_SPLITS": NUM_SPLITS,
        "BLOCK_DI": BLOCK_DI,
        "BLOCK_DJ": BLOCK_DJ,
        "KV_BLOCK_M": KV_BLOCK_M,
        "OUT_BLOCK_M": OUT_BLOCK_M,
        "OUT_BLOCK_D": OUT_BLOCK_D,
        "BLOCK_K": BLOCK_K,
        "CHUNK_TILES": chunk_tiles,
        "MMA_PRECISION": "tf32x3_manual",
        "PHI_occupancy": PHI_OCCUPANCY,
        "PARTIAL_occupancy": PARTIAL_OCCUPANCY,
        "REDUCE_occupancy": REDUCE_OCCUPANCY,
        "OUT_occupancy": OUT_OCCUPANCY,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
