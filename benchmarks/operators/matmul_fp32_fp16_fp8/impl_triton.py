import torch
import triton
import triton.language as tl


# Per-dtype default config (used when autotune=False).
_DEFAULT_CONFIGS = {
    torch.float32: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
    },
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
    },
    torch.float8_e5m2: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3,
    },
}


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Generic GEMM with grouped scheduling. fp32 accumulator; output dtype is
    inferred from c_ptr — fp32, fp16, fp8 e4m3fn, and fp8 e5m2 are all handled
    in the epilogue."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # input_precision="tf32" enables TF32 acceleration when inputs are fp32;
        # ignored for fp16 / fp8 inputs, so this is safe to set unconditionally.
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast accumulator to the output dtype.
    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    elif c_ptr.dtype.element_ty == tl.float8e5:
        c = accumulator.to(tl.float8e5)
    elif c_ptr.dtype.element_ty == tl.float32:
        c = accumulator
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Single autotune wrapper covering all dtypes. Triton recompiles per-dtype
# automatically (c_ptr.dtype.element_ty changes the epilogue path).
matmul_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gs,
            },
            num_warps=nw,
            num_stages=ns,
        )
        # Shrunk from 216 cfgs to 16 to match cuTile-side shrink (impl_cutile.py).
        for bm in [128, 256]
        for bn in [128, 256]
        for bk in [64, 128]
        for gs in [8]
        for nw in [4, 8]
        for ns in [3]
    ],
    key=["M", "N", "K"],
    warmup=1,
    rep=3,
)(matmul_kernel)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    """Triton matmul. Output dtype matches input dtype (fp32 / fp16 / fp8)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if autotune:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        matmul_kernel_autotuned[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    else:
        if a.dtype not in _DEFAULT_CONFIGS:
            raise ValueError(f"No default config for dtype {a.dtype}")
        cfg = _DEFAULT_CONFIGS[a.dtype]
        grid = (
            triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
        )
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return c


def get_last_config() -> dict | None:
    cfg = getattr(matmul_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_M": cfg.kwargs["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": cfg.kwargs["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": cfg.kwargs["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": cfg.kwargs["GROUP_SIZE_M"],
        "num_warps":    cfg.num_warps,
        "num_stages":   cfg.num_stages,
    }
