import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


# fp32 kernel: manual TF32x3 — split each fp32 input into hi (tfloat32) and
# lo (tfloat32 of the residual), then do 3 tensor-core MMAs per K-tile
# (a_hi*b_hi + a_hi*b_lo + a_lo*b_hi). Drops a_lo*b_lo (negligible).
# This recovers fp32-grade precision while running on TF32 tensor cores
# (~1100 TFLOPS) instead of the 60 TFLOPS IEEE fp32 ALU path.
@ct.kernel(occupancy=1)
def _matmul_kernel_tf32x3(a, b, c,
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

        a_hi = ct.astype(a_tile, ct.tfloat32)
        a_hi_f = ct.astype(a_hi, np.float32)
        a_lo = ct.astype(a_tile - a_hi_f, ct.tfloat32)

        b_hi = ct.astype(b_tile, ct.tfloat32)
        b_hi_f = ct.astype(b_hi, np.float32)
        b_lo = ct.astype(b_tile - b_hi_f, ct.tfloat32)

        acc = ct.mma(a_lo, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)
        acc = ct.mma(a_hi, b_hi, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


# fp16 / fp8 kernel: straight MMA, large tile, occ=1 (iter 6 best config).
@ct.kernel(occupancy=1)
def _matmul_kernel_lowp(a, b, c,
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
        # Manual TF32x3 — preserves fp32 precision but uses TF32 tensor cores.
        TM, TN, TK = 128, 128, 32
        kernel = _matmul_kernel_tf32x3
        occupancy = 1
        precision = "tf32x3"
    else:
        TM, TN, TK = 128, 256, 64
        kernel = _matmul_kernel_lowp
        occupancy = 1
        precision = "default"

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": occupancy, "precision": precision})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
