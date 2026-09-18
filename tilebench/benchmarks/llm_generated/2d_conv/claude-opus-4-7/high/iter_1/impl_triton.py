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
    PRECISION: tl.constexpr,
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

        # weight viewed as (OC, K)
        weight_offs = offs_n[None, :] * K + k[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(weight_ptr + weight_offs, mask=w_mask, other=0.0)

        acc = tl.dot(a, w, acc, input_precision=PRECISION)

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

    if input.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        GROUP_M = 8
        num_warps = 4
        num_stages = 2
        PRECISION = "ieee"
    else:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        GROUP_M = 8
        num_warps = 4
        num_stages = 3
        PRECISION = "tf32"

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _conv2d_kernel[grid](
        input, weight, output,
        batch, IC, OC, H, W, OH, OW, M, N, K,
        KS=KS, STRIDE=stride, PADDING=padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        PRECISION=PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M, "num_warps": num_warps, "num_stages": num_stages,
        "PRECISION": PRECISION,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
