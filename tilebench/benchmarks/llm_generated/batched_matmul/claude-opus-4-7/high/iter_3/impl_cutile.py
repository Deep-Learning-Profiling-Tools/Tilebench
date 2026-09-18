import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _bmm_kernel_lp(A, B, C,
                   M: ConstInt, N: ConstInt, K: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 path: single MMA per K-tile."""
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
        acc = ct.mma(a, b, acc)

    out = ct.astype(acc, A.dtype).reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


@ct.kernel(occupancy=1)
def _bmm_kernel_fp32(A, B, C,
                     M: ConstInt, N: ConstInt, K: ConstInt,
                     TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp32 path: TF32x3 emulation via Veltkamp split (a = a_hi + a_lo
    where a_hi has only ~10 mantissa bits, so the TF32 tensor cores see
    a_hi/b_hi/a_lo/b_lo losslessly). 3 MMAs per K-tile approximate IEEE
    fp32 precision."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    batch = ct.bid(2)

    # Veltkamp split constant: C = 2^13 + 1 splits fp32 (23-bit mantissa)
    # into hi (10-bit, matching TF32) + lo residual.
    SPLIT_C = 8193.0

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a = ct.load(A, index=(batch, pid_m, k),
                    shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TM, TK))
        b = ct.load(B, index=(batch, k, pid_n),
                    shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TK, TN))

        # Split a into (a_hi, a_lo); both fp32 but a_hi has TF32 precision.
        ca = SPLIT_C * a
        a_hi = ca - (ca - a)
        a_lo = a - a_hi

        cb = SPLIT_C * b
        b_hi = cb - (cb - b)
        b_lo = b - b_hi

        # 3 MMAs (drops a_lo * b_lo term which is ~2^-20 relative).
        acc = ct.mma(a_hi, b_hi, acc)
        acc = ct.mma(a_lo, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)

    out = acc.reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    TM = 128
    TN = 128
    TK = 32
    is_fp32 = (A.dtype == torch.float32)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)
    if is_fp32:
        ct.launch(stream, grid, _bmm_kernel_fp32,
                  (A3, B3, C3, M, N, K, TM, TN, TK))
    else:
        ct.launch(stream, grid, _bmm_kernel_lp,
                  (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": 1 if is_fp32 else 2,
                      "IS_FP32": is_fp32})
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
