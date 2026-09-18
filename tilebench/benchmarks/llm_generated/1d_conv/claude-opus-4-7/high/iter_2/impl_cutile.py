import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_kernel(x, w, out, input_size, output_size,
                   BLOCK_M: ConstInt,
                   K_PAD: ConstInt,
                   BLOCK_N: ConstInt):
    bid = ct.bid(0)
    base = bid * BLOCK_M * BLOCK_N

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)
    offs_k = ct.arange(K_PAD, dtype=np.int32)
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)

    # X[m, k] = input[base + m*BLOCK_N + k]
    x_indices = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    X = ct.gather(x, x_indices, padding_value=0.0)

    # W[k, n] = kernel[k - n] if 0 <= k - n < kernel_size else 0
    # ct.gather treats negative or >= size indices as OOB -> padding_value
    k_minus_n = offs_k[:, None] - offs_n[None, :]
    W = ct.gather(w, k_minus_n, padding_value=0.0)

    # Tensor-core mma: acc = X @ W
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    acc = ct.mma(X, W, acc)

    # Store: out[base + m*BN + n] = acc[m, n]
    out_indices = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_indices, ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    BLOCK_M = 64
    K_PAD = 256
    BLOCK_N = 16
    occupancy = 2

    grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
    ct.launch(stream, grid, _conv1d_kernel,
              (input, kernel, output, input_size, output_size,
               BLOCK_M, K_PAD, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
        "occupancy": occupancy,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
