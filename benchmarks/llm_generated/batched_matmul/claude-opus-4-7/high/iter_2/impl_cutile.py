import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _bmm_kernel(A, B, C,
                M: ConstInt, N: ConstInt, K: ConstInt,
                TM: ConstInt, TN: ConstInt, TK: ConstInt,
                IS_FP32: ConstBool):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    batch = ct.bid(2)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a = ct.load(A, index=(batch, pid_m, k),
                    shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TM, TK))
        b = ct.load(B, index=(batch, k, pid_n),
                    shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TK, TN))
        if IS_FP32:
            # TF32x3 emulation: split fp32 into TF32 hi+lo, do 3 MMAs.
            # Gives ~IEEE fp32 precision via Tensor Cores.
            a_hi = ct.astype(a, ct.tfloat32)
            b_hi = ct.astype(b, ct.tfloat32)
            a_lo = ct.astype(a - ct.astype(a_hi, np.float32), ct.tfloat32)
            b_lo = ct.astype(b - ct.astype(b_hi, np.float32), ct.tfloat32)
            acc = ct.mma(a_hi, b_hi, acc)
            acc = ct.mma(a_lo, b_hi, acc)
            acc = ct.mma(a_hi, b_lo, acc)
        else:
            acc = ct.mma(a, b, acc)

    out = ct.astype(acc, A.dtype).reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    TM = 128
    TN = 128
    TK = 32
    IS_FP32 = (A.dtype == torch.float32)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)
    ct.launch(stream, grid, _bmm_kernel,
              (A3, B3, C3, M, N, K, TM, TN, TK, IS_FP32))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": 2, "IS_FP32": IS_FP32})
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
