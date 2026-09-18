```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _phi2_kernel(Q, K, PhiQ, PhiK, N,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    q = tl.load(Q + offs, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K + offs, mask=mask, other=0.0).to(tl.float32)

    phi_q = tl.where(q > 0.0, q + 1.0, tl.exp(q))
    phi_k = tl.where(k > 0.0, k + 1.0, tl.exp(k))

    tl.store(PhiQ + offs, phi_q, mask=mask)
    tl.store(PhiK + offs, phi_k, mask=mask)


@triton.jit
def _partial_kv_kernel(PhiK, V, partial_S, partial_Z, M,
                       D: tl.constexpr,
                       CHUNK_TILES: tl.constexpr,
                       BLOCK_DI: tl.constexpr,
                       BLOCK_DJ: tl.constexpr,
                       KV_BLOCK_M: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_i = pid_i * BLOCK_DI + tl.arange(0, BLOCK_DI)
    offs_j = pid_j * BLOCK_DJ + tl.arange(0, BLOCK_DJ)
    offs_m = tl.arange(0, KV_BLOCK_M)

    acc = tl.zeros((BLOCK_DI, BLOCK_DJ), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_DI,), dtype=tl.float32)

    for t in tl.static_range(0, CHUNK_TILES):
        row_tile = pid_s * CHUNK_TILES + t
        rows = row_tile * KV_BLOCK_M + offs_m

        k_mask = (rows[:, None] < M) & (offs_i[None, :] < D)
        v_mask = (rows[:, None] < M) & (offs_j[None, :] < D)

        k_phi = tl.load(
            PhiK + rows[:, None] * D + offs_i[None, :],
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        v_tile = tl.load(
            V + rows[:, None] * D + offs_j[None, :],
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(tl.trans(k_phi), v_tile, acc,
                     input_precision="tf32x3", out_dtype=tl.float32)
        z_acc += tl.sum(k_phi, axis=0)

    s_mask = (offs_i[:, None] < D) & (offs_j[None, :] < D)
    tl.store(
        partial_S + pid_s * D * D + offs_i[:, None] * D + offs_j[None, :],
        acc,
        mask=s_mask,
    )

    tl.store(
        partial_Z + pid_s * D + offs_i,
        z_acc,
        mask=(offs_i < D) & (pid_j == 0),
    )


@triton.jit
def _reduce_kv_kernel(partial_S, partial_Z, S, Z,
                      D: tl.constexpr,
                      NUM_SPLITS: tl.constexpr,
                      BLOCK_DI: tl.constexpr,
                      BLOCK_DJ: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_DI + tl.arange(0, BLOCK_DI)
    offs_j = pid_j * BLOCK_DJ + tl.arange(0, BLOCK_DJ)

    acc = tl.zeros((BLOCK_DI, BLOCK_DJ), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_DI,), dtype=tl.float32)

    mask_s = (offs_i[:, None] < D) & (offs_j[None, :] < D)
    mask_z = offs_i < D

    for s in tl.range(0, NUM_SPLITS, 1, num_stages=1):
        ps = tl.load(
            partial_S + s * D * D + offs_i[:, None] * D + offs_j[None, :],
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        acc += ps

        pz = tl.load(
            partial_Z + s * D + offs_i,
            mask=mask_z,
            other=0.0,
        ).to(tl.float32)
        z_acc += pz

    tl.store(S + offs_i[:, None] * D + offs_j[None, :], acc, mask=mask_s)
    tl.store(Z + offs_i, z_acc, mask=mask_z & (pid_j == 0))


@triton.jit
def _output_kernel(PhiQ, S, Z, O, M,
                   D: tl.constexpr,
                   eps,
                   OUT_BLOCK_M: tl.constexpr,
                   OUT_BLOCK_D: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * OUT_BLOCK_M + tl.arange(0, OUT_BLOCK_M)
    offs_n = pid_n * OUT_BLOCK_D + tl.arange(0, OUT_BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((OUT_BLOCK_M, OUT_BLOCK_D), dtype=tl.float32)
    denom = tl.zeros((OUT_BLOCK_M,), dtype=tl.float32)

    for k0 in tl.static_range(0, D, BLOCK_K):
        kidx = k0 + offs_k

        q_phi = tl.load(
            PhiQ + offs_m[:, None] * D + kidx[None, :],
            mask=(offs_m[:, None] < M) & (kidx[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        s_tile = tl.load(
            S + kidx[:, None] * D + offs_n[None, :],
            mask=(kidx[:, None] < D) & (offs_n[None, :] < D),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        acc = tl.dot(q_phi, s_tile, acc,
                     input_precision="tf32x3", out_dtype=tl.float32)

        z_tile = tl.load(
            Z + kidx,
            mask=kidx < D,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        denom += tl.sum(q_phi * z_tile[None, :], axis=1)

    out = acc / (denom[:, None] + eps)
    tl.store(
        O + offs_m[:, None] * D + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < D),
    )


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M = Q.shape[0]
    D = Q.shape[1]
    N = M * D

    output = torch.empty_like(Q)

    PHI_BLOCK_SIZE = 1024
    NUM_SPLITS = 32
    BLOCK_DI = 64
    BLOCK_DJ = 128
    KV_BLOCK_M = 64
    OUT_BLOCK_M = 64
    OUT_BLOCK_D = 128
    BLOCK_K = 64

    PHI_NUM_WARPS = 4
    PHI_NUM_STAGES = 2
    KV_NUM_WARPS = 4
    KV_NUM_STAGES = 3
    REDUCE_NUM_WARPS = 4
    REDUCE_NUM_STAGES = 1
    OUT_NUM_WARPS = 4
    OUT_NUM_STAGES = 3

    row_tiles = triton.cdiv(M, KV_BLOCK_M)
    chunk_tiles = triton.cdiv(row_tiles, NUM_SPLITS)
    d_tiles_i = triton.cdiv(D, BLOCK_DI)
    d_tiles_j = triton.cdiv(D, BLOCK_DJ)
    out_m_tiles = triton.cdiv(M, OUT_BLOCK_M)
    out_n_tiles = triton.cdiv(D, OUT_BLOCK_D)

    PhiQ = torch.empty_like(Q)
    PhiK = torch.empty_like(K)
    partial_S = torch.empty((NUM_SPLITS, D, D), device=Q.device, dtype=torch.float32)
    partial_Z = torch.empty((NUM_SPLITS, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    _phi2_kernel[(triton.cdiv(N, PHI_BLOCK_SIZE),)](
        Q, K, PhiQ, PhiK, N,
        BLOCK_SIZE=PHI_BLOCK_SIZE,
        num_warps=PHI_NUM_WARPS,
        num_stages=PHI_NUM_STAGES,
    )

    _partial_kv_kernel[(d_tiles_i, d_tiles_j, NUM_SPLITS)](
        PhiK, V, partial_S, partial_Z, M,
        D=D,
        CHUNK_TILES=chunk_tiles,
        BLOCK_DI=BLOCK_DI,
        BLOCK_DJ=BLOCK_DJ,
        KV_BLOCK_M=KV_BLOCK_M,
        num_warps=KV_NUM_WARPS,
        num_stages=KV_NUM_STAGES,
    )

    _reduce_kv_kernel[(d_tiles_i, d_tiles_j)](
        partial_S, partial_Z, S, Z,
        D=D,
        NUM_SPLITS=NUM_SPLITS,
        BLOCK_DI=BLOCK_DI,
        BLOCK_DJ=BLOCK_DJ,
        num_warps=REDUCE_NUM_WARPS,
        num_stages=REDUCE_NUM_STAGES,
    )

    _output_kernel[(out_m_tiles, out_n_tiles)](
        PhiQ, S, Z, output, M,
        D=D,
        eps=float(eps),
        OUT_BLOCK_M=OUT_BLOCK_M,
        OUT_BLOCK_D=OUT_BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=OUT_NUM_WARPS,
        num_stages=OUT_NUM_STAGES,
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
        "DOT_PRECISION": "tf32x3",
        "PHI_num_warps": PHI_NUM_WARPS,
        "PHI_num_stages": PHI_NUM_STAGES,
        "KV_num_warps": KV_NUM_WARPS,
        "KV_num_stages": KV_NUM_STAGES,
        "REDUCE_num_warps": REDUCE_NUM_WARPS,
        "REDUCE_num_stages": REDUCE_NUM_STAGES,
        "OUT_num_warps": OUT_NUM_WARPS,
        "OUT_num_stages": OUT_NUM_STAGES,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
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


@ct.kernel(occupancy=2)
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

        acc = ct.mma(ct.transpose(k_phi), v_tile, acc)
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


@ct.kernel(occupancy=2)
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


@ct.kernel(occupancy=2)
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
    N = M * D

    output = torch.empty_like(Q)

    PHI_BLOCK_SIZE = 1024
    NUM_SPLITS = 32
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
        _phi2_kernel,
        (Q.reshape(-1), K.reshape(-1), PhiQ.reshape(-1), PhiK.reshape(-1),
         N, PHI_BLOCK_SIZE),
    )

    ct.launch(
        stream,
        (d_tiles_i, d_tiles_j, NUM_SPLITS),
        _partial_kv_kernel,
        (PhiK, V, partial_S, partial_Z, M, D,
         chunk_tiles, BLOCK_DI, BLOCK_DJ, KV_BLOCK_M),
    )

    ct.launch(
        stream,
        (d_tiles_i, d_tiles_j, 1),
        _reduce_kv_kernel,
        (partial_S, partial_Z, S, Z, NUM_SPLITS, BLOCK_DI, BLOCK_DJ),
    )

    ct.launch(
        stream,
        (out_m_tiles, out_n_tiles, 1),
        _output_kernel,
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
        "PHI_occupancy": PHI_OCCUPANCY,
        "PARTIAL_occupancy": PARTIAL_OCCUPANCY,
        "REDUCE_occupancy": REDUCE_OCCUPANCY,
        "OUT_occupancy": OUT_OCCUPANCY,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
