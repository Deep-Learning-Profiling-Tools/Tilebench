import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    total_tiles, num_pid_m, num_pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_start = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    iters_per_tile = tl.cdiv(K, BLOCK_K)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(num_pid_m > 0)
    tl.assume(num_pid_n > 0)

    for tile_id in tl.range(pid_start, total_tiles, num_progs, num_stages=1):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(iters_per_tile):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if a.dtype == torch.float32:
        # IEEE fp32 path — uses CUDA cores, matches cuBLAS sgemm bit-for-bit
        # within fp32 default tolerance. Smaller tile/shallow pipeline keeps
        # the CUDA-core mainloop register-light.
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        num_warps = 8
        num_stages = 2
        input_precision = "ieee"
    else:
        # fp16 / bf16 — tensor cores with TF32 acc semantics.
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        num_warps = 8
        num_stages = 3
        input_precision = "tf32"

    GROUP_SIZE_M = 8

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_ctas = min(NUM_SMS, total_tiles)
    if num_ctas <= 0:
        num_ctas = 1

    grid = (num_ctas,)
    _matmul_persistent_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        total_tiles, num_pid_m, num_pid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps, "num_stages": num_stages,
        "num_ctas": num_ctas,
        "input_precision": input_precision,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
