import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _build_b_kernel(kernel, bmat,
                    KERNEL_SIZE: ConstInt,
                    BLOCK_N: ConstInt,
                    BUILD_M: ConstInt,
                    BUILD_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    rows = bid_m * BUILD_M + ct.arange(BUILD_M, dtype=np.int32)[:, None]
    cols = bid_n * BUILD_N + ct.arange(BUILD_N, dtype=np.int32)[None, :]

    j = rows - cols
    vals = ct.gather(kernel, j, padding_value=0.0, check_bounds=True, latency=1)

    ct.store(
        bmat,
        index=(bid_m, bid_n),
        tile=ct.astype(vals, bmat.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _conv1d_mma_kernel(input, bmat, output,
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
        kk = (k0 + k_1d)[None, :]
        a_idx = rows * BLOCK_N + kk
        a = ct.gather(input, a_idx, padding_value=0.0, check_bounds=True, latency=1)

        b = ct.load(
            bmat,
            index=(k0 // BLOCK_K, 0),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        )

        acc = ct.mma(a, b, acc)

    out_idx = rows * BLOCK_N + cols
    ct.scatter(output, out_idx, ct.astype(acc, input.dtype), check_bounds=True, latency=1)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    K_TOTAL_PAD = ((kernel_size + BLOCK_N - 1 + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    DIRECT_BLOCK = BLOCK_M * BLOCK_N

    BUILD_M = BLOCK_K
    BUILD_N = BLOCK_N

    occupancy_build = 4
    occupancy = 1

    bmat = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=input.dtype)

    build_grid = (
        (K_TOTAL_PAD + BUILD_M - 1) // BUILD_M,
        (BLOCK_N + BUILD_N - 1) // BUILD_N,
        1,
    )
    build_kernel = _build_b_kernel.with_hints(occupancy=occupancy_build)
    ct.launch(
        stream,
        build_grid,
        build_kernel,
        (kernel, bmat, kernel_size, BLOCK_N, BUILD_M, BUILD_N),
    )

    grid = ((output_size + DIRECT_BLOCK - 1) // DIRECT_BLOCK, 1, 1)
    conv_kernel = _conv1d_mma_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        conv_kernel,
        (input, bmat, output, K_TOTAL_PAD, BLOCK_M, BLOCK_N, BLOCK_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "DIRECT_BLOCK": DIRECT_BLOCK,
        "BUILD_M": BUILD_M,
        "BUILD_N": BUILD_N,
        "occupancy_build": occupancy_build,
        "occupancy": occupancy,
        "DENSE_B": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
