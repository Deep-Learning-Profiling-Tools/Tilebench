Looking at the trajectory, iter 1 is only at 1.6% roofline despite ~4.7× torch speedup. The bottleneck is that we issue 127 sequential load+FMA pairs per output element, hitting only L1 throughput rather than HBM. To reach high roofline, we need **Tensor Cores**.

**Key insight**: Reformulate 1D conv as a matmul where the N dimension produces *useful* outputs (not just padded zeros). If row m,col n of the output block corresponds to `output[base + m*BLOCK_N + n]`, then with `W[k,n] = kernel[k-n]` (zero where out of range), we get `(X @ W)[m,n] = output[base + m*BN + n]` where `X[m,k] = input[base + m*BN + k]`. This is a true im2col→GEMM. With `BLOCK_N=16` we get 16 useful outputs per row instead of wasting 15/16 of TC compute on padding.

Config: `BLOCK_M=64, K_PAD=256, BLOCK_N=16`. K_PAD=256 covers `K+BN-1=142`. Per block produces 1024 outputs with input L1 amplification of ~1.23× (near-optimal HBM ratio).

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                   KERNEL_SIZE: tl.constexpr,
                   BLOCK_M: tl.constexpr,
                   K_PAD: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_M * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K_PAD)
    offs_n = tl.arange(0, BLOCK_N)

    # X[m, k] = input[base + m*BLOCK_N + k]
    x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    x_mask = x_addrs < input_size
    X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

    # W[k, n] = kernel[k - n] if 0 <= k - n < KERNEL_SIZE else 0
    k_minus_n = offs_k[:, None] - offs_n[None, :]
    w_valid = (k_minus_n >= 0) & (k_minus_n < KERNEL_SIZE)
    w_idx = tl.where(w_valid, k_minus_n, 0)
    W = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)

    # Tensor-core matmul: acc = X @ W
    acc = tl.dot(X, W, out_dtype=tl.float32)  # [BLOCK_M, BLOCK_N]

    # Store: output[base + m*BN + n] = acc[m, n]
    out_addrs = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_addrs < output_size
    tl.store(out_ptr + out_addrs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    BLOCK_M = 64
    K_PAD = 256
    BLOCK_N = 16
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(output_size, BLOCK_M * BLOCK_N),)
    _conv1d_kernel[grid](
        input, kernel, output, input_size, output_size,
        KERNEL_SIZE=kernel_size,
        BLOCK_M=BLOCK_M, K_PAD=K_PAD, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
```

Reformulated 1D conv as a Tensor-Core GEMM via a shifted-kernel im2col matrix `W[k,n]=kernel[k-n]` so all 16 columns of the matmul produce real outputs (not padding waste). Each program computes `BLOCK_M*BLOCK_N=1024` outputs with input L1 amplification ≈ 1.23×, which should dramatically improve over the sequential FMA loop.
