import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _conv1d_kernel(input, kernel, output,
                   K_TOTAL_PAD: ConstInt,
                   BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    bid = ct.bid(0)

    rows_1d = bid * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    cols_1d = ct.arange(BLOCK_N, dtype=np.int32)
    k_1d = ct.arange(BLOCK_K, dtype=np.int32)

    rows = rows_1d[:, None]
    cols = cols_1d[None, :]

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        kk_row = (k0 + k_1d)[None, :]
        a_idx = rows * BLOCK_N + kk_row
        a = ct.gather(input, a_idx, padding_value=0.0, check_bounds=True)

        kk_col = (k0 + k_1d)[:, None]
        b_idx = kk_col - cols
        b = ct.gather(kernel, b_idx, padding_value=0.0, check_bounds=True)

        acc = ct.mma(a, b, acc)

    out_idx = rows * BLOCK_N + cols
    ct.scatter(output, out_idx, ct.astype(acc, input.dtype), check_bounds=True)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    K_TOTAL_PAD = ((kernel_size + BLOCK_N - 1 + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    occupancy = 2

    num_row_groups = (output_size + BLOCK_N - 1) // BLOCK_N
    grid = ((num_row_groups + BLOCK_M - 1) // BLOCK_M, 1, 1)

    launch_kernel = _conv1d_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        launch_kernel,
        (input, kernel, output, K_TOTAL_PAD, BLOCK_M, BLOCK_N, BLOCK_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
