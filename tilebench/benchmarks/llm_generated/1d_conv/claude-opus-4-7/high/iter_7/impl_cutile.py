import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_matmul_kernel(x, w, out, input_size, output_size,
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

        x_idx = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        X = ct.gather(x, x_idx, padding_value=0.0)

        k_minus_n = offs_k[:, None] - offs_n[None, :]
        W = ct.gather(w, k_minus_n, padding_value=0.0)

        acc = ct.mma(X, W, acc)

    out_idx = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_idx, ct.astype(acc, out.dtype))


@ct.kernel(occupancy=4)
def _conv1d_scalar_kernel(x, w, out, output_size,
                          KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE
    offs = base + ct.arange(TILE, dtype=np.int32)

    acc = ct.full((TILE,), 0.0, dtype=np.float32)
    for j in range(KERNEL_SIZE):
        xj = ct.gather(x, offs + j, padding_value=0.0)
        wj = ct.load(w, index=(j,), shape=())
        acc = acc + ct.astype(xj, np.float32) * ct.astype(wj, np.float32)

    ct.scatter(out, offs, ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if input.dtype in (torch.float16, torch.bfloat16):
        # MMA path — Tensor Cores with fp32 accumulator
        BLOCK_M = 64
        BLOCK_N = 256
        BLOCK_K = 64                       # bumped 32 → 64 for higher MMA throughput
        # ceil((BLOCK_N + KERNEL_SIZE - 1) / BLOCK_K) = ceil(382/64) = 6
        NUM_K_ITERS = (BLOCK_N + kernel_size - 1 + BLOCK_K - 1) // BLOCK_K
        occupancy = 2

        grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
        ct.launch(stream, grid, _conv1d_matmul_kernel,
                  (input, kernel, output, input_size, output_size,
                   BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_ITERS, kernel_size))

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
            "NUM_K_ITERS": NUM_K_ITERS,
            "occupancy": occupancy,
        })
    else:
        TILE = 2048
        occupancy = 4
        grid = (ct.cdiv(output_size, TILE), 1, 1)
        ct.launch(stream, grid, _conv1d_scalar_kernel,
                  (input, kernel, output, output_size, kernel_size, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "scalar",
            "TILE": TILE,
            "occupancy": occupancy,
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
