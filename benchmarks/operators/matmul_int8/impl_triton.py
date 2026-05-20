import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 4,
}


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Int8 GEMM with 2-bit packed B (4 fields per byte).

    Outer loop walks K_b = K/4 in BLOCK_SIZE_K tiles, loading each B byte tile
    from HBM exactly once. The inner unrolled `for i in range(4)` loop unpacks
    the i-th 2-bit field in registers and pairs it with the corresponding
    A tile at K offset `i*K_b + j*BLOCK_SIZE_K`. Accumulator is int32 via
    tl.dot's IMMA path.
    """
    tl.static_assert(
        K % (4 * BLOCK_SIZE_K) == 0,
        "K must be divisible by 4*BLOCK_SIZE_K (so K_b is divisible by BLOCK_SIZE_K)",
    )
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    one_i8 = tl.full((1,), 1, dtype=tl.int8)
    K_b: tl.constexpr = K // 4
    num_kb_tiles = tl.cdiv(K_b, BLOCK_SIZE_K)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K_b, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    for j in range(0, num_kb_tiles):
        packed_k = j * BLOCK_SIZE_K
        b_uint8 = b_desc.load([packed_k, start_n])
        for i in range(4):
            k_pos = i * K_b + packed_k
            a = a_desc.load([start_m, k_pos]).to(tl.int8)
            mask_i = 3 << (2 * i)
            b = ((b_uint8 & mask_i) >> (2 * i)).to(tl.int8)
            accumulator += tl.dot(a, b - one_i8, out_dtype=tl.int32)

    c_desc.store([start_m, start_n], accumulator)


matmul_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
            },
            num_warps=nw,
            num_stages=ns,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [32, 64]
        for nw in [4, 8, 16]
        for ns in [3, 4]
        if bm * bn <= 128 * 256
    ],
    key=["M", "N", "K"],
    warmup=3,
    rep=10,
)(matmul_kernel)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    """Triton int8 GEMM with 2-bit packed B."""
    ensure_tma_available()
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    assert a.shape[1] == b.shape[0] * 4, (
        "Incompatible dims: A's K must equal 4 * B's K_b (B is packed 4-per-byte)"
    )
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

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
        cfg = _DEFAULT_CONFIG
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
