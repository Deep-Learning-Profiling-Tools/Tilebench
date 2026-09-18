import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _batched_matmul_kernel(
    A,
    B,
    C,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    USE_FP32_PRECISE: ConstBool,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_b = ct.bid(2)

    num_k_tiles = ct.cdiv(K, TILE_K)
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

    if USE_FP32_PRECISE:
        for k_tile in range(0, num_k_tiles):
            a_tile_3d = ct.load(
                A,
                index=(bid_b, bid_m, k_tile),
                shape=(1, TILE_M, TILE_K),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_tile_3d = ct.load(
                B,
                index=(bid_b, k_tile, bid_n),
                shape=(1, TILE_K, TILE_N),
                padding_mode=ct.PaddingMode.ZERO,
            )

            a_f = ct.astype(ct.reshape(a_tile_3d, (TILE_M, TILE_K)), np.float32)
            b_f = ct.astype(ct.reshape(b_tile_3d, (TILE_K, TILE_N)), np.float32)

            a0 = ct.astype(a_f, ct.float16)
            b0 = ct.astype(b_f, ct.float16)
            a0_f = ct.astype(a0, np.float32)
            b0_f = ct.astype(b0, np.float32)

            ar1 = a_f - a0_f
            br1 = b_f - b0_f
            a1 = ct.astype(ar1, ct.float16)
            b1 = ct.astype(br1, ct.float16)
            a1_f = ct.astype(a1, np.float32)
            b1_f = ct.astype(b1, np.float32)

            ar2 = ar1 - a1_f
            br2 = br1 - b1_f
            a2 = ct.astype(ar2, ct.float16)
            b2 = ct.astype(br2, ct.float16)

            acc = ct.mma(a0, b0, acc)

            acc = ct.mma(a0, b1, acc)
            acc = ct.mma(a1, b0, acc)

            acc = ct.mma(a0, b2, acc)
            acc = ct.mma(a1, b1, acc)
            acc = ct.mma(a2, b0, acc)

            acc = ct.mma(a1, b2, acc)
            acc = ct.mma(a2, b1, acc)
            acc = ct.mma(a2, b2, acc)
    else:
        for k_tile in range(0, num_k_tiles):
            a_tile_3d = ct.load(
                A,
                index=(bid_b, bid_m, k_tile),
                shape=(1, TILE_M, TILE_K),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_tile_3d = ct.load(
                B,
                index=(bid_b, k_tile, bid_n),
                shape=(1, TILE_K, TILE_N),
                padding_mode=ct.PaddingMode.ZERO,
            )

            a_tile = ct.reshape(a_tile_3d, (TILE_M, TILE_K))
            b_tile = ct.reshape(b_tile_3d, (TILE_K, TILE_N))
            acc = ct.mma(a_tile, b_tile, acc)

    out_tile = ct.astype(acc, A.dtype)
    out_tile_3d = ct.reshape(out_tile, (1, TILE_M, TILE_N))
    ct.store(C, index=(bid_b, bid_m, bid_n), tile=out_tile_3d)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A_3d = A.view(BATCH, M, K)
    B_3d = B.view(BATCH, K, N)
    C_3d = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

    TILE_M = 64
    TILE_N = 128
    TILE_K = 32
    occupancy = 2
    USE_FP32_PRECISE = (A.dtype == torch.float32)

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), BATCH)
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        grid,
        _batched_matmul_kernel,
        (A_3d, B_3d, C_3d, K, TILE_M, TILE_N, TILE_K, USE_FP32_PRECISE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE_M": TILE_M,
        "TILE_N": TILE_N,
        "TILE_K": TILE_K,
        "occupancy": occupancy,
        "fp32_precision": "fp16_3chunk_9term",
        "USE_FP32_PRECISE": USE_FP32_PRECISE,
    })
    return C_3d.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
