import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _batched_matmul_kernel(
    A,
    B,
    C,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_b = ct.bid(2)

    num_k_tiles = ct.cdiv(K, TILE_K)

    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

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

    TILE_M = 128
    TILE_N = 64
    TILE_K = 64
    occupancy = 2

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), BATCH)
    stream = torch.cuda.current_stream()

    kernel = _batched_matmul_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (A_3d, B_3d, C_3d, K, TILE_M, TILE_N, TILE_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE_M": TILE_M,
        "TILE_N": TILE_N,
        "TILE_K": TILE_K,
        "occupancy": occupancy,
    })
    return C_3d.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
