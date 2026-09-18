import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_kernel(x, w, out, input_size, output_size,
                   BLOCK_M: ConstInt,
                   BLOCK_N: ConstInt,
                   BLOCK_K: ConstInt,
                   NUM_K_ITERS: ConstInt,
                   KERNEL_SIZE: ConstInt):
    bid = ct.bid(0)
    base = bid * BLOCK_M * BLOCK_N

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for kk in range(NUM_K_ITERS):
        offs_k = ct.arange(BLOCK_K, dtype=np.int32) + (kk * BLOCK_K)

        # X[m, k] = input[base + m*BLOCK_N + k]
        x_idx = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        X = ct.gather(x, x_idx, padding_value=0.0)

        # W[k, n] = kernel[k - n]   (gather returns padding for negative idx)
        k_minus_n = offs_k[:, None] - offs_n[None, :]
        W = ct.gather(w, k_minus_n, padding_value=0.0)

        acc = ct.mma(X, W, acc)

    out_idx = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_idx, ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    # BLOCK_N (=256) > KERNEL_SIZE-1 (=126) → rows non-overlapping in input,
    # minimising memory amplification. 12 K-tiles of 32 cover 382 needed.
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32
    total_k = BLOCK_N + kernel_size - 1
    NUM_K_ITERS = (total_k + BLOCK_K - 1) // BLOCK_K
    occupancy = 2

    grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
    ct.launch(stream, grid, _conv1d_kernel,
              (input, kernel, output, input_size, output_size,
               BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_ITERS, kernel_size))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "NUM_K_ITERS": NUM_K_ITERS,
        "occupancy": occupancy,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
