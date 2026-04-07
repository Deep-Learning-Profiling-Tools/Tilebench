import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}


@triton.jit
def _quantized_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    out_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        k_offs = k_start * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < m) & (k_offs[None, :] < k)
        b_mask = (k_offs[:, None] < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.float32)
        acc += tl.dot(a, b)

    out = acc * out_scale
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, out, mask=c_mask)


_quantized_gemm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [32, 64]
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["m", "n", "k"],
)(_quantized_gemm_kernel)


def run(a_q: torch.Tensor, b_q: torch.Tensor, scale: float, block_size: int = 1024, autotune: bool = False, **kwargs):
    if a_q.dim() != 2 or b_q.dim() != 2:
        raise ValueError("matmul_int8 expects 2D inputs.")
    if a_q.shape[1] != b_q.shape[0]:
        raise ValueError("Inner dimensions must match for GEMM.")

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    m, k = a_q.shape
    _, n = b_q.shape
    out = torch.empty((m, n), device=a_q.device, dtype=torch.float32)

    if autotune:
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
        _quantized_gemm_kernel_autotuned[grid](
            a_q, b_q, out,
            m, n, k,
            a_q.stride(0), a_q.stride(1),
            b_q.stride(0), b_q.stride(1),
            out.stride(0), out.stride(1),
            scale * scale,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))
        _quantized_gemm_kernel[grid](
            a_q, b_q, out,
            m, n, k,
            a_q.stride(0), a_q.stride(1),
            b_q.stride(0), b_q.stride(1),
            out.stride(0), out.stride(1),
            scale * scale,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_quantized_gemm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M":    cfg.kwargs["BLOCK_M"],
        "BLOCK_N":    cfg.kwargs["BLOCK_N"],
        "BLOCK_K":    cfg.kwargs["BLOCK_K"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
