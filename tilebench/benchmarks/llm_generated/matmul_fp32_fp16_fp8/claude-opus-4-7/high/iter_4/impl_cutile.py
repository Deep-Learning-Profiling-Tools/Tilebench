import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _matmul_kernel_occ2(a, b, c,
                        K: ConstInt,
                        TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a_tile = ct.load(a, index=(bid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


@ct.kernel(occupancy=1)
def _matmul_kernel_occ1(a, b, c,
                        K: ConstInt,
                        TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a_tile = ct.load(a, index=(bid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    stream = torch.cuda.current_stream()

    if a.dtype == torch.float32:
        # fp32: try TK=64 with occupancy=1 (2x compute per K-iter vs iter 3's
        # TK=32 occupancy=2). Shmem for fp32 128x64 + 64x128 double-buffered
        # = 128KB, fits at occ=1. If verify fails next iter rolls back.
        TM, TN, TK = 128, 128, 64
        kernel = _matmul_kernel_occ1
        occupancy = 1
    else:
        # fp16 / fp8: iter 2's fastest config.
        TM, TN, TK = 128, 256, 64
        kernel = _matmul_kernel_occ1
        occupancy = 1

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
