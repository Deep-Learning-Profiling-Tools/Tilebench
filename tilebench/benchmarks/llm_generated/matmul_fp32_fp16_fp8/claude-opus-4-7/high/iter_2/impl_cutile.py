import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _matmul_kernel_tf32(a, b, c,
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
        TM, TN, TK = 128, 128, 32
        kernel = _matmul_kernel_tf32
    else:
        TM, TN, TK = 128, 256, 64
        kernel = _matmul_kernel_lp

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK, "occupancy": 1})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
