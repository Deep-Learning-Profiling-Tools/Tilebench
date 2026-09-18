Looking at the trajectory:
- **Triton iter 7 (2.9%) regressed** from iter 2 (3.0%) — revert to iter 2's clean baseline.
- **Cutile iter 7 (14.3%) was a big win for fp16** but broke fp32 verify because `ct.mma` uses TF32. The fix: keep iter 7's fast unified mma kernel for fp16, and use iter 6's manual outer-product path for fp32.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

# Force PyTorch reference to use IEEE fp32 (matches our IEEE-precision kernel)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

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
    offs_k_base = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k_base
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

        weight_offs = offs_n[None, :] * K + k[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(weight_ptr + weight_offs, mask=w_mask, other=0.0)

        acc = tl.dot(a, w, acc, input_precision="ieee")

    out_offs = (b[:, None] * (OC * OHW)
                + offs_n[None, :] * OHW
                + oh[:, None] * OW
                + ow[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offs,
             acc.to(output_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, weight, stride=1, padding=1, groups=1, **kwargs):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

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

    # Revert to iter-2 baseline (best verify-clean for triton).
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
        "PRECISION": "ieee",
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


# ---------------------------------------------------------------------------
# fp16 path: ct.mma-based (iter 7's winner — 14.3% / 19.6x).
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=2)
def _conv2d_kernel_mma(
    input_flat, weight_3d, output_flat,
    batch, H, W, OH, OW, M,
    IC: ConstInt, OC: ConstInt, KS: ConstInt,
    STRIDE: ConstInt, PADDING: ConstInt,
    NUM_C_BLOCKS: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_C: ConstInt,
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)

    OHW = OH * OW
    HW = H * W
    ICHW = IC * HW

    b = offs_m // OHW
    rem_m = offs_m % OHW
    oh = rem_m // OW
    ow = rem_m % OW

    b_base = b * ICHW

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    valid_m = offs_m < M
    valid_n = offs_n < OC

    for kh in range(0, KS):
        for kw in range(0, KS):
            ih = oh * STRIDE + kh - PADDING
            iw = ow * STRIDE + kw - PADDING
            in_bounds_m = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid_m
            ihw = ih * W + iw

            kk_idx = kh * KS + kw

            for c_blk in range(0, NUM_C_BLOCKS):
                ic = c_blk * BLOCK_C + ct.arange(BLOCK_C, dtype=np.int32)
                ic_valid = ic < IC

                input_idx = b_base[:, None] + ihw[:, None] + ic[None, :] * HW
                valid_full = in_bounds_m[:, None] & ic_valid[None, :]
                input_idx_safe = ct.where(valid_full, input_idx, -1)
                a = ct.gather(input_flat, input_idx_safe, padding_value=0.0)

                w_tile = ct.load(weight_3d, index=(pid_n, kk_idx, c_blk),
                                 shape=(BLOCK_N, 1, BLOCK_C),
                                 padding_mode=ct.PaddingMode.ZERO)
                w_tile = ct.reshape(w_tile, (BLOCK_N, BLOCK_C))
                w_t = ct.transpose(w_tile)
                acc = ct.mma(a, w_t, acc)

    out_idx = (b[:, None] * (OC * OHW)
               + offs_n[None, :] * OHW
               + oh[:, None] * OW
               + ow[:, None])
    valid_out = valid_m[:, None] & valid_n[None, :]
    out_idx_safe = ct.where(valid_out, out_idx, -1)
    result = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_idx_safe, result)


# ---------------------------------------------------------------------------
# fp32 path: manual outer-product to avoid TF32 precision loss in ct.mma.
# Iter 6 verified-clean.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=2)
def _conv2d_kernel_manual_fp32(
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

        valid_m = offs_m < M
        valid_k = offs_k < K_CONST
        in_bounds = ((ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                     & valid_m[:, None] & valid_k[None, :])

        input_idx = (b[:, None] * ICHW
                     + ic[None, :] * HW
                     + ih * W
                     + iw)
        input_idx_safe = ct.where(in_bounds, input_idx, -1)

        a = ct.gather(input_flat, input_idx_safe, padding_value=0.0)
        w_tile = ct.load(weight_2d, index=(pid_n, k_block),
                         shape=(BLOCK_N, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        w_t = ct.transpose(w_tile)

        for kk in range(BLOCK_K):
            a_col = ct.extract(a, (0, kk), shape=(BLOCK_M, 1))
            w_row = ct.extract(w_t, (kk, 0), shape=(1, BLOCK_N))
            acc = acc + a_col * w_row

    out_idx = (b[:, None] * (OC * OHW)
               + offs_n[None, :] * OHW
               + oh[:, None] * OW
               + ow[:, None])
    valid_out = (offs_m < M)[:, None] & (offs_n < OC)[None, :]
    out_idx_safe = ct.where(valid_out, out_idx, -1)

    result = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_idx_safe, result)


def run(input, weight, stride=1, padding=1, groups=1, **kwargs):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

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
    output_flat = output.view(-1)

    stream = torch.cuda.current_stream()

    if input.dtype == torch.float32:
        # Manual outer-product path — preserves IEEE fp32 accuracy.
        weight_2d = weight_c.view(OC, K)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(OC, BLOCK_N), 1)
        ct.launch(stream, grid, _conv2d_kernel_manual_fp32, (
            input_flat, weight_2d, output_flat,
            batch, H, W, OH, OW, M,
            IC, OC, KS, K, stride, padding,
            BLOCK_M, BLOCK_N, BLOCK_K,
        ))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
            "occupancy": 2, "path": "manual_fp32",
        })
    else:
        # fp16: ct.mma path — iter 7's winner. BLOCK_C=64, NUM_C_BLOCKS=2.
        weight_perm = weight_c.permute(0, 2, 3, 1).contiguous()
        weight_3d = weight_perm.view(OC, KS * KS, IC)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_C = 64
        NUM_C_BLOCKS = (IC + BLOCK_C - 1) // BLOCK_C
        grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(OC, BLOCK_N), 1)
        ct.launch(stream, grid, _conv2d_kernel_mma, (
            input_flat, weight_3d, output_flat,
            batch, H, W, OH, OW, M,
            IC, OC, KS, stride, padding,
            NUM_C_BLOCKS,
            BLOCK_M, BLOCK_N, BLOCK_C,
        ))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_C": BLOCK_C,
            "NUM_C_BLOCKS": NUM_C_BLOCKS,
            "occupancy": 2, "path": "mma_fp16_v2",
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Strategy: revert Triton to iter-2's verify-clean baseline (3.0%, no regression). For cuTile, combine iter-7's fast mma kernel (14.3% on fp16) with iter-6's manual outer-product kernel for fp32 (verify-clean), branched on dtype — recovering both wins without breaking fp32 verification.
