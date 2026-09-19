import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_matmul_kernel(x, w, out, input_size, output_size,
                          BLOCK_M: ConstInt,
                          K_PAD: ConstInt,
                          BLOCK_N: ConstInt):
    bid = ct.bid(0)
    base = bid * BLOCK_M * BLOCK_N

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)
    offs_k = ct.arange(K_PAD, dtype=np.int32)
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)

    x_indices = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    X = ct.gather(x, x_indices, padding_value=0.0)

    k_minus_n = offs_k[:, None] - offs_n[None, :]
    W = ct.gather(w, k_minus_n, padding_value=0.0)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    acc = ct.mma(X, W, acc)

    out_indices = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_indices, ct.astype(acc, out.dtype))


@ct.kernel(occupancy=4)
def _conv1d_scalar_kernel(x, w, out, input_size, output_size,
                          KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    t1 = ct.load(x, index=(bid,), shape=(TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    t2 = ct.load(x, index=(bid + 1,), shape=(TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    big = ct.cat((t1, t2), axis=0)
    big_f32 = ct.astype(big, np.float32)

    acc = ct.full((TILE,), 0.0, dtype=np.float32)
    for j in range(KERNEL_SIZE):
        sub = ct.extract(big_f32, (j,), shape=(TILE,))
        wj = ct.load(w, index=(j,), shape=())
        acc = acc + sub * ct.astype(wj, np.float32)

    ct.store(out, index=(bid,), tile=ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if input.dtype in (torch.float16, torch.bfloat16):
        BLOCK_M = 64
        K_PAD = 256
        BLOCK_N = 16
        grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
        ct.launch(stream, grid, _conv1d_matmul_kernel,
                  (input, kernel, output, input_size, output_size,
                   BLOCK_M, K_PAD, BLOCK_N))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
            "occupancy": 2,
        })
    else:
        TILE = 2048
        grid = (ct.cdiv(output_size, TILE), 1, 1)
        ct.launch(stream, grid, _conv1d_scalar_kernel,
                  (input, kernel, output, input_size, output_size,
                   kernel_size, TILE))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "scalar",
            "TILE": TILE,
            "occupancy": 4,
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
