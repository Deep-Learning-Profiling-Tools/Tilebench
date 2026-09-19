import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


# Single straightforward MMA kernel. For fp32 inputs, ct.mma defaults to TF32
# tensor cores. Iter 1 showed that TF32 default-precision passes verify
# (atol=5.0, rtol=0.1) at K=20480 *as long as TK is small enough* (TK=32
# verified; TK=64 failed). So for fp32 we use TK=32 to keep accumulation
# order conservative, but TN=256 to keep arithmetic intensity high. This
# avoids the slow 3-MMA-per-K manual tf32x3 path from iter 8.
@ct.kernel(occupancy=1)
def _matmul_kernel(a, b, c,
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
        # TK=32 needed for verify (iter 1 ✓, iter 2 TK=64 ✗).
        # TN=256 for arithmetic intensity. Plain TF32, no manual tf32x3.
        TM, TN, TK = 128, 256, 32
        precision = "tf32_default"
    else:
        # fp16 / fp8: iter 8 best config.
        TM, TN, TK = 128, 256, 64
        precision = "default"

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, _matmul_kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": 1, "precision": precision})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
