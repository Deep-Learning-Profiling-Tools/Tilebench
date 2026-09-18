import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _partial_kv_kernel(K, V, partial_S, partial_Z, M, D,
                       CHUNK_TILES: tl.constexpr,
                       NUM_SPLITS: tl.constexpr,
                       BLOCK_D: tl.constexpr,
                       KV_BLOCK_M: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_i = pid_i * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_j = pid_j * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, KV_BLOCK_M)

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in tl.static_range(0, CHUNK_TILES):
        row_tile = pid_s * CHUNK_TILES + t
        rows = row_tile * KV_BLOCK_M + offs_m

        k_mask = (rows[:, None] < M) & (offs_i[None, :] < D)
        v_mask = (rows[:, None] < M) & (offs_j[None, :] < D)

        k_raw = tl.load(
            K + rows[:, None] * D + offs_i[None, :],
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        k_phi_val = tl.where(k_raw > 0.0, k_raw + 1.0, tl.exp(k_raw))
        k_phi = tl.where(k_mask, k_phi_val, 0.0)

        v_tile = tl.load(
            V + rows[:, None] * D + offs_j[None, :],
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(tl.trans(k_phi), v_tile, acc, input_precision="tf32x3", out_dtype=tl.float32)

        if pid_j == 0:
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
def _reduce_kv_kernel(partial_S, partial_Z, S, Z, D,
                      NUM_SPLITS: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_j = pid_j * BLOCK_D + tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    mask_s = (offs_i[:, None] < D) & (offs_j[None, :] < D)
    mask_z = offs_i < D

    for s in tl.range(0, NUM_SPLITS, 1, num_stages=1):
        ps = tl.load(
            partial_S + s * D * D + offs_i[:, None] * D + offs_j[None, :],
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        acc += ps

        if pid_j == 0:
            pz = tl.load(
                partial_Z + s * D + offs_i,
                mask=mask_z,
                other=0.0,
            ).to(tl.float32)
            z_acc += pz

    tl.store(S + offs_i[:, None] * D + offs_j[None, :], acc, mask=mask_s)
    tl.store(Z + offs_i, z_acc, mask=mask_z & (pid_j == 0))


@triton.jit
def _output_kernel(Q, S, Z, O, M, D, eps,
                   OUT_BLOCK_M: tl.constexpr,
                   BLOCK_D: tl.constexpr,
                   BLOCK_K: tl.constexpr,
                   OUT_NUM_STAGES: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * OUT_BLOCK_M + tl.arange(0, OUT_BLOCK_M)
    offs_n = pid_n * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((OUT_BLOCK_M, BLOCK_D), dtype=tl.float32)
    denom = tl.zeros((OUT_BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, D, BLOCK_K, num_stages=OUT_NUM_STAGES):
        kidx = k0 + offs_k

        q_mask = (offs_m[:, None] < M) & (kidx[None, :] < D)
        q_raw = tl.load(
            Q + offs_m[:, None] * D + kidx[None, :],
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_phi_val = tl.where(q_raw > 0.0, q_raw + 1.0, tl.exp(q_raw))
        q_phi = tl.where(q_mask, q_phi_val, 0.0)

        s_tile = tl.load(
            S + kidx[:, None] * D + offs_n[None, :],
            mask=(kidx[:, None] < D) & (offs_n[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(q_phi, s_tile, acc, input_precision="tf32x3", out_dtype=tl.float32)

        z_tile = tl.load(Z + kidx, mask=kidx < D, other=0.0).to(tl.float32)
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

    output = torch.empty_like(Q)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)

    NUM_SPLITS = 40
    BLOCK_D = 64
    KV_BLOCK_M = 64
    OUT_BLOCK_M = 64
    BLOCK_K = 64

    KV_NUM_WARPS = 4
    KV_NUM_STAGES = 3
    REDUCE_NUM_WARPS = 4
    REDUCE_NUM_STAGES = 1
    OUT_NUM_WARPS = 4
    OUT_NUM_STAGES = 3

    row_tiles = triton.cdiv(M, KV_BLOCK_M)
    chunk_tiles = triton.cdiv(row_tiles, NUM_SPLITS)
    d_tiles = triton.cdiv(D, BLOCK_D)

    partial_S = torch.empty((NUM_SPLITS, D, D), device=Q.device, dtype=torch.float32)
    partial_Z = torch.empty((NUM_SPLITS, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    _partial_kv_kernel[(d_tiles, d_tiles, NUM_SPLITS)](
        K, V, partial_S, partial_Z, M, D,
        CHUNK_TILES=chunk_tiles,
        NUM_SPLITS=NUM_SPLITS,
        BLOCK_D=BLOCK_D,
        KV_BLOCK_M=KV_BLOCK_M,
        num_warps=KV_NUM_WARPS,
        num_stages=KV_NUM_STAGES,
    )

    _reduce_kv_kernel[(d_tiles, d_tiles)](
        partial_S, partial_Z, S, Z, D,
        NUM_SPLITS=NUM_SPLITS,
        BLOCK_D=BLOCK_D,
        num_warps=REDUCE_NUM_WARPS,
        num_stages=REDUCE_NUM_STAGES,
    )

    _output_kernel[(triton.cdiv(M, OUT_BLOCK_M), d_tiles)](
        Q, S, Z, output, M, D, float(eps),
        OUT_BLOCK_M=OUT_BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        OUT_NUM_STAGES=OUT_NUM_STAGES,
        num_warps=OUT_NUM_WARPS,
        num_stages=OUT_NUM_STAGES,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "NUM_SPLITS": NUM_SPLITS,
        "BLOCK_D": BLOCK_D,
        "KV_BLOCK_M": KV_BLOCK_M,
        "OUT_BLOCK_M": OUT_BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "CHUNK_TILES": chunk_tiles,
        "DOT_PRECISION": "tf32x3",
        "Z_GUARD": True,
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
