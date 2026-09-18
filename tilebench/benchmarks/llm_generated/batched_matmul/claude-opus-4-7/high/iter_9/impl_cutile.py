import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _bmm_kernel(A, B, C,
                M: ConstInt, N: ConstInt, K: ConstInt,
                TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 batched matmul via tensor cores with fp32 accumulator."""
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


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    # ct.mma routes fp32 through TF32 cores (~11 mantissa bits). At K=640
    # accumulated error exceeds verify tolerance for fp32. Skip fp32.
    if A.dtype == torch.float32:
        raise NotImplementedError(
            "cuTile ct.mma cannot reach IEEE fp32 precision (TF32-cores only)."
        )

    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    stream = torch.cuda.current_stream()

    # Iter 9: bump occupancy from 2 to 4 to improve SM utilization.
    # Per-CTA shmem with TK=32 is small (A: 8KB, B: 16KB), so higher
    # occupancy should fit and improve latency hiding.
    TM, TN, TK = 128, 256, 32
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)

    ct.launch(stream, grid, _bmm_kernel,
              (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TK": TK,
        "occupancy": 4,
        "fp32_path": "skip",
    })
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
