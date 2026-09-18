import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _partial_kv_kernel(K, V, partial_S, partial_Z,
                       M: ConstInt, D: ConstInt,
                       CHUNK_TILES: ConstInt,
                       BLOCK_D: ConstInt,
                       KV_BLOCK_M: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)
    pid_s = ct.bid(2)

    offs_m = ct.arange(KV_BLOCK_M, dtype=np.int32)[:, None]
    offs_i = ct.arange(BLOCK_D, dtype=np.int32)[None, :]
    offs_j = ct.arange(BLOCK_D, dtype=np.int32)[None, :]

    cols_i = pid_i * BLOCK_D + offs_i
    cols_j = pid_j * BLOCK_D + offs_j

    acc = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)

    for t in range(0, CHUNK_TILES):
        row_tile = pid_s * CHUNK_TILES + t
        rows = row_tile * KV_BLOCK_M + offs_m

        k_raw = ct.astype(
            ct.load(
                K,
                index=(row_tile, pid_i),
                shape=(KV_BLOCK_M, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )
        v_tile = ct.astype(
            ct.load(
                V,
                index=(row_tile, pid_j),
                shape=(KV_BLOCK_M, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        mask_k = (rows < M) & (cols_i < D)
        mask_v = (rows < M) & (cols_j < D)

        k_phi_val = ct.where(k_raw > 0.0, k_raw + 1.0, ct.exp(k_raw))
        k_phi = ct.where(mask_k, k_phi_val, 0.0)
        v_tile = ct.where(mask_v, v_tile, 0.0)

        acc = ct.mma(ct.transpose(k_phi), v_tile, acc)
        z_acc = z_acc + ct.sum(k_phi, axis=0)

    ct.store(
        partial_S,
        index=(pid_s, pid_i, pid_j),
        tile=ct.reshape(acc, (1, BLOCK_D, BLOCK_D)),
    )

    if pid_j == 0:
        ct.store(
            partial_Z,
            index=(pid_s, pid_i),
            tile=ct.reshape(z_acc, (1, BLOCK_D)),
        )


@ct.kernel(occupancy=4)
def _reduce_kv_kernel(partial_S, partial_Z, S, Z,
                      NUM_SPLITS: ConstInt,
                      BLOCK_D: ConstInt):
    pid_i = ct.bid(0)
    pid_j = ct.bid(1)

    acc = ct.full((BLOCK_D, BLOCK_D), 0.0, dtype=np.float32)
    z_acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)

    for s in range(0, NUM_SPLITS):
        ps = ct.load(
            partial_S,
            index=(s, pid_i, pid_j),
            shape=(1, BLOCK_D, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.reshape(ct.astype(ps, np.float32), (BLOCK_D, BLOCK_D))

        pz = ct.load(
            partial_Z,
            index=(s, pid_i),
            shape=(1, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        z_acc = z_acc + ct.reshape(ct.astype(pz, np.float32), (BLOCK_D,))

    ct.store(S, index=(pid_i, pid_j), tile=acc)

    if pid_j == 0:
        ct.store(Z, index=(pid_i,), tile=z_acc)


@ct.kernel(occupancy=4)
def _output_kernel(Q, S, Z, output, eps,
                   M: ConstInt, D: ConstInt,
                   NUM_K_TILES: ConstInt,
                   OUT_BLOCK_M: ConstInt,
                   BLOCK_D: ConstInt,
                   BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    row_offsets = pid_m * OUT_BLOCK_M + ct.arange(OUT_BLOCK_M, dtype=np.int32)[:, None]

    acc = ct.full((OUT_BLOCK_M, BLOCK_D), 0.0, dtype=np.float32)
    denom = ct.full((OUT_BLOCK_M,), 0.0, dtype=np.float32)

    for kb in range(0, NUM_K_TILES):
        k_offsets = kb * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)[None, :]

        q_raw = ct.astype(
            ct.load(
                Q,
                index=(pid_m, kb),
                shape=(OUT_BLOCK_M, BLOCK_K),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        q_mask = (row_offsets < M) & (k_offsets < D)
        q_phi_val = ct.where(q_raw > 0.0, q_raw + 1.0, ct.exp(q_raw))
        q_phi = ct.where(q_mask, q_phi_val, 0.0)

        s_tile = ct.astype(
            ct.load(
                S,
                index=(kb, pid_n),
                shape=(BLOCK_K, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        acc = ct.mma(q_phi, s_tile, acc)

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

    output = torch.empty_like(Q)

    NUM_SPLITS = 32
    BLOCK_D = 64
    KV_BLOCK_M = 64
    OUT_BLOCK_M = 64
    BLOCK_K = 64

    PARTIAL_OCCUPANCY = 4
    REDUCE_OCCUPANCY = 4
    OUT_OCCUPANCY = 4

    row_tiles = (M + KV_BLOCK_M - 1) // KV_BLOCK_M
    chunk_tiles = (row_tiles + NUM_SPLITS - 1) // NUM_SPLITS
    d_tiles = (D + BLOCK_D - 1) // BLOCK_D
    num_k_tiles = (D + BLOCK_K - 1) // BLOCK_K

    partial_S = torch.empty((NUM_SPLITS, D, D), device=Q.device, dtype=torch.float32)
    partial_Z = torch.empty((NUM_SPLITS, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (d_tiles, d_tiles, NUM_SPLITS),
        _partial_kv_kernel,
        (K, V, partial_S, partial_Z, M, D, chunk_tiles, BLOCK_D, KV_BLOCK_M),
    )

    ct.launch(
        stream,
        (d_tiles, d_tiles, 1),
        _reduce_kv_kernel,
        (partial_S, partial_Z, S, Z, NUM_SPLITS, BLOCK_D),
    )

    ct.launch(
        stream,
        ((M + OUT_BLOCK_M - 1) // OUT_BLOCK_M, d_tiles, 1),
        _output_kernel,
        (Q, S, Z, output, float(eps), M, D, num_k_tiles, OUT_BLOCK_M, BLOCK_D, BLOCK_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "NUM_SPLITS": NUM_SPLITS,
        "BLOCK_D": BLOCK_D,
        "KV_BLOCK_M": KV_BLOCK_M,
        "OUT_BLOCK_M": OUT_BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "CHUNK_TILES": chunk_tiles,
        "PARTIAL_occupancy": PARTIAL_OCCUPANCY,
        "REDUCE_occupancy": REDUCE_OCCUPANCY,
        "OUT_occupancy": OUT_OCCUPANCY,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
