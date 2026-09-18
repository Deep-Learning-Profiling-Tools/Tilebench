import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# Mask that clears the lower 13 mantissa bits of an fp32 → produces a value
# representable exactly in TF32 (10-bit mantissa). -8192 == 0xFFFFE000 in int32.
_TF32_MASK = -8192


@ct.kernel(occupancy=2)
def _bmm_kernel_lp(A, B, C,
                   M: ConstInt, N: ConstInt, K: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 path: fp32 accumulator via ct.mma."""
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


@ct.kernel(occupancy=2)
def _bmm_kernel_fp32_3xtf32(A, B, C,
                            M: ConstInt, N: ConstInt, K: ConstInt,
                            TM: ConstInt, TN: ConstInt, TK: ConstInt,
                            MASK: ConstInt):
    """fp32 path: 3xTF32 (Markidis) emulation to recover ~fp32 precision.
       Each input is split as x = x_hi + x_lo where x_hi is rounded to TF32
       and x_lo is the residual. Then C += A_hi*B_hi + A_lo*B_hi + A_hi*B_lo
       (drop A_lo*B_lo as it's ~2^-26 of the result). Each ct.mma uses TF32
       cores, giving aggregate precision close to true fp32 matmul."""
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

        # Truncate to TF32 precision via bitmask, capture residual.
        a_int = ct.bitcast(a, np.int32)
        a_hi = ct.bitcast(a_int & MASK, np.float32)
        a_lo = a - a_hi
        b_int = ct.bitcast(b, np.int32)
        b_hi = ct.bitcast(b_int & MASK, np.float32)
        b_lo = b - b_hi

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
    is_fp32 = (A.dtype == torch.float32)

    stream = torch.cuda.current_stream()

    TM, TN, TK = 128, 128, 32
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)

    if is_fp32:
        ct.launch(stream, grid, _bmm_kernel_fp32_3xtf32,
                  (A3, B3, C3, M, N, K, TM, TN, TK, _TF32_MASK))
    else:
        ct.launch(stream, grid, _bmm_kernel_lp,
                  (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TK": TK,
        "occupancy": 2,
        "FP32_EMU": "3xTF32" if is_fp32 else "none",
    })
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
