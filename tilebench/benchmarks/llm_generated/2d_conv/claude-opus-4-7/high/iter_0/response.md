Looking at this 2D convolution problem, I'll use **implicit GEMM**: treat conv as a matmul where M = batch·OH·OW, N = out_channels, K = (in_channels/groups)·KS·KS. Each output position's K-dim values come from input via padded indirection (im2col on the fly).

For the test config (batch=1, C=128, K=3×3, padding=1, stride=1): M=H², N=128, K=1152. Tensor-core-friendly tile of 128×128×32 fp16/fp32.

- **Triton**: standard matmul loop, compute input addresses on the fly with bounds masking, weight as plain 2D (OC × K) view.
- **cuTile**: same approach but use `ct.gather` for input (irregular padded access) with negative-index masking for OOB, `ct.load` for weight tile + transpose, `ct.scatter` for output.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, IC, OC, H, W, OH, OW, M, N, K,
    KS: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    OHW = OH * OW
    b = offs_m // OHW
    rem_m = offs_m % OHW
    oh = rem_m // OW
    ow = rem_m % OW

    m_mask = offs_m < M
    n_mask = offs_n < N

    KSKS = KS * KS
    HW = H * W
    ICHW = IC * HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        k_mask = k < K

        ic = k // KSKS
        krem = k % KSKS
        kh = krem // KS
        kw = krem % KS

        ih = oh[:, None] * STRIDE + kh[None, :] - PADDING
        iw = ow[:, None] * STRIDE + kw[None, :] - PADDING

        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        input_offs = (b[:, None] * ICHW
                      + ic[None, :] * HW
                      + ih * W
                      + iw)
        a_mask = m_mask[:, None] & k_mask[None, :] & in_bounds
        a = tl.load(input_ptr + input_offs, mask=a_mask, other=0.0)

        # weight viewed as (OC, K); element [oc, k] at offset oc*K + k
        weight_offs = offs_n[None, :] * K + k[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(weight_ptr + weight_offs, mask=w_mask, other=0.0)

        acc = tl.dot(a, w, acc)

    out_offs = (b[:, None] * (OC * OHW)
                + offs_n[None, :] * OHW
                + oh[:, None] * OW
                + ow[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offs,
             acc.to(output_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, weight, stride=1, padding=1, groups=1, **kwargs):
    batch, IC, H, W = input.shape
    OC = weight.shape[0]
    KS = weight.shape[2]
    OH = (H + 2 * padding - KS) // stride + 1
    OW = (W + 2 * padding - KS) // stride + 1

    output = torch.empty((batch, OC, OH, OW), dtype=input.dtype, device=input.device)

    M = batch * OH * OW
    N = OC
    K = (IC // groups) * KS * KS

    input = input.contiguous()
    weight = weight.contiguous()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _conv2d_kernel[grid](
        input, weight, output,
        batch, IC, OC, H, W, OH, OW, M, N, K,
        KS=KS, STRIDE=stride, PADDING=padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M, "num_warps": num_warps, "num_stages": num_stages,
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


@ct.kernel
def _conv2d_kernel(
    input_flat, weight_2d, output_flat,
    batch, H, W, OH, OW, M,
    IC: ConstInt, OC: ConstInt, KS: ConstInt, K_CONST: ConstInt,
    STRIDE: ConstInt, PADDING: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt,
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)

    OHW = OH * OW
    HW = H * W
    ICHW = IC * HW
    OC_OHW = OC * OHW

    b = offs_m // OHW
    rem_m = offs_m % OHW
    oh = rem_m // OW
    ow = rem_m % OW

    KSKS = KS * KS
    num_k_blocks = ct.cdiv(K_CONST, BLOCK_K)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for k_block in range(0, num_k_blocks):
        offs_k = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)

        ic = offs_k // KSKS
        krem = offs_k % KSKS
        kh = krem // KS
        kw = krem % KS

        ih = oh[:, None] * STRIDE + kh[None, :] - PADDING
        iw = ow[:, None] * STRIDE + kw[None, :] - PADDING

        valid_k = offs_k < K_CONST
        valid_m = offs_m < M
        in_bounds = ((ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                     & valid_m[:, None] & valid_k[None, :])

        input_idx = (b[:, None] * ICHW
                     + ic[None, :] * HW
                     + ih * W
                     + iw)
        input_idx = ct.where(in_bounds, input_idx, -1)

        a = ct.gather(input_flat, input_idx, padding_value=0.0)

        w_tile = ct.load(weight_2d, index=(pid_n, k_block),
                         shape=(BLOCK_N, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        w_t = ct.transpose(w_tile)

        acc = ct.mma(a, w_t, acc)

    out_idx = (b[:, None] * OC_OHW
               + offs_n[None, :] * OHW
               + oh[:, None] * OW
               + ow[:, None])
    valid_m = offs_m < M
    valid_n = offs_n < OC
    valid = valid_m[:, None] & valid_n[None, :]
    out_idx = ct.where(valid, out_idx, -1)

    result = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_idx, result)


def run(input, weight, stride=1, padding=1, groups=1, **kwargs):
    batch, IC, H, W = input.shape
    OC = weight.shape[0]
    KS = weight.shape[2]
    OH = (H + 2 * padding - KS) // stride + 1
    OW = (W + 2 * padding - KS) // stride + 1

    output = torch.empty((batch, OC, OH, OW), dtype=input.dtype, device=input.device)

    M = batch * OH * OW
    K = (IC // groups) * KS * KS

    input_c = input.contiguous()
    weight_c = weight.contiguous()

    input_flat = input_c.view(-1)
    weight_2d = weight_c.view(OC, K)
    output_flat = output.view(-1)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    occupancy = 2

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(OC, BLOCK_N), 1)

    stream = torch.cuda.current_stream()
    kernel = _conv2d_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (
        input_flat, weight_2d, output_flat,
        batch, H, W, OH, OW, M,
        IC, OC, KS, K, stride, padding,
        BLOCK_M, BLOCK_N, BLOCK_K,
    ))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "occupancy": occupancy,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```
