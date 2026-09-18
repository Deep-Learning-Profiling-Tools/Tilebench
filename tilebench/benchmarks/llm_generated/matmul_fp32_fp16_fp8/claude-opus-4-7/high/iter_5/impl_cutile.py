import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


# fp32 kernel: explicit tfloat32 cast forces TF32 tensor-core path.
@ct.kernel(occupancy=2)
def _matmul_kernel_fp32(a, b, c,
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
        a_tf = ct.astype(a_tile, ct.tfloat32)
        b_tf = ct.astype(b_tile, ct.tfloat32)
        acc = ct.mma(a_tf, b_tf, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


# fp16 / fp8 kernel: no cast, direct mma.
@ct.kernel(occupancy=1)
def _matmul_kernel_lp(a, b, c,
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
        # fp32: iter 1/3 verified config (TK=32, occ=2), now with explicit
        # tfloat32 cast to ensure tensor-core dispatch.
        TM, TN, TK = 128, 128, 32
        kernel = _matmul_kernel_fp32
        occupancy = 2
    else:
        # fp16 / fp8: iter 2's fast config.
        TM, TN, TK = 128, 256, 64
        kernel = _matmul_kernel_lp
        occupancy = 1

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
