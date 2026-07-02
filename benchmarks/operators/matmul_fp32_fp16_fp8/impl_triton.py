import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Per-dtype default config (used when autotune=False). Chosen from the TMA
# kernel's autotune winners at sweep-max K: fp32 needs the small-K deep
# pipeline (4-byte tiles are smem-limited); fp16/fp8 want big 256x256 tiles.
_DEFAULT_CONFIGS = {
    torch.float32: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
    },
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
    },
}


def _tma_set_block_size_hook(nargs):
    """Autotune pre-hook: point the host-side TMA descriptors at the config's
    block shape before each trial launch (pattern from Triton tutorial
    09-persistent-matmul; host-side TensorDescriptor is the supported TMA API —
    device-side tl.make_tensor_descriptor is currently broken)."""
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]   # B passed transposed (N, K)
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


# TMA wants both GEMM operands' inner (contiguous) box dim to be the K axis
# (box width ≤ 128B hits the TMA swizzle fast path; a [BLOCK_K, BLOCK_N] box on
# row-major (K, N) B has a 256–512B inner dim and measurably degrades fp16/fp8).
# So, like the tutorial, the kernel consumes B transposed to (N, K). Build the
# transposed copy once per input tensor (keyed by data_ptr) so the unmeasured
# warmup call pays for it — same caching pattern as impl_torch._fp8_args.
_bt_cache: dict = {}


def _b_transposed(b: torch.Tensor) -> torch.Tensor:
    key = (b.data_ptr(), b.shape, b.dtype)
    if key not in _bt_cache:
        _bt_cache[key] = b.t().contiguous()   # (N, K) row-major
    return _bt_cache[key]


@triton.jit
def matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    # TensorDescriptor args are not torch.Tensors, so the autotuner does NOT
    # fold their dtype into its cache key like it does for tensor args. DT_ID
    # (0=fp32, 1=fp16, 2=fp8e4m3) stands in for the dtype so each dtype tunes
    # its own winner instead of inheriting the first dtype's config.
    DT_ID: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Generic GEMM with grouped scheduling and host-side TMA loads/stores
    (non-persistent — one CTA per output tile, matching impl_cutile.py's
    grid). fp32 accumulator; output dtype follows c_desc — fp32, fp16 and
    fp8 e4m3fn are all handled by the epilogue cast."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(tl.cdiv(K, BLOCK_SIZE_K)):
        # TMA tile loads; b_desc views B transposed (N, K), so both boxes have
        # the K axis innermost (≤128B ⇒ TMA swizzle fast path).
        a = a_desc.load([offs_am, k * BLOCK_SIZE_K])
        b = b_desc.load([offs_bn, k * BLOCK_SIZE_K])
        # input_precision="tf32" enables TF32 acceleration when inputs are fp32;
        # ignored for fp16 / fp8 inputs, so this is safe to set unconditionally.
        accumulator = tl.dot(a, b.T, accumulator, input_precision="tf32")

    # Cast accumulator to the output dtype (from the descriptor) and TMA-store.
    c_desc.store([offs_am, offs_bn], accumulator.to(c_desc.dtype))


# Single autotune wrapper covering all dtypes. Triton recompiles per-dtype
# automatically (the descriptor dtype changes the epilogue path).
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
            pre_hook=_tma_set_block_size_hook,
        )
        # Shrunk from 216 cfgs to 16 to match cuTile-side shrink (impl_cutile.py).
        for bm in [128, 256]
        for bn in [128, 256]
        for bk in [64, 128]
        for gs in [8]
        for nw in [4, 8]
        for ns in [3]
    ] + [
        # fp32 needs these: with 4-byte elements + TMA store staging, every
        # ns=3 config above exceeds B200's 227KB smem (min 262KB) and Triton's
        # autotuner would pick an OutOfResources config. 128x128x64 @ ns=2 is
        # 192KB and fits. cuTile needs no analog — its exhaustive_search
        # auto-skips compiler-rejected configs.
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": bk,
             "GROUP_SIZE_M": 8},
            num_warps=nw,
            num_stages=ns,
            pre_hook=_tma_set_block_size_hook,
        )
        for bk, ns in [(64, 2), (32, 3), (32, 4)]
        for nw in [4, 8]
    ],
    key=["M", "N", "K", "DT_ID"],
    warmup=1,
    rep=3,
)(matmul_kernel)


_DT_IDS = {torch.float32: 0, torch.float16: 1, torch.float8_e4m3fn: 2}


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    """Triton matmul. Output dtype matches input dtype (fp32 / fp16 / fp8)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    bt = _b_transposed(b)   # (N, K); cached, so warmup pays the copy

    if autotune:
        # Dummy block shape — the autotune pre_hook overwrites it per config.
        dummy_block = [1, 1]
        a_desc = TensorDescriptor.from_tensor(a, dummy_block)
        b_desc = TensorDescriptor.from_tensor(bt, dummy_block)
        c_desc = TensorDescriptor.from_tensor(c, dummy_block)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        matmul_kernel_autotuned[grid](a_desc, b_desc, c_desc, M, N, K,
                                      DT_ID=_DT_IDS[a.dtype])
    else:
        if a.dtype not in _DEFAULT_CONFIGS:
            raise ValueError(f"No default config for dtype {a.dtype}")
        cfg = _DEFAULT_CONFIGS[a.dtype]
        bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
        a_desc = TensorDescriptor.from_tensor(a, [bm, bk])
        b_desc = TensorDescriptor.from_tensor(bt, [bn, bk])
        c_desc = TensorDescriptor.from_tensor(c, [bm, bn])
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        matmul_kernel[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
            DT_ID=_DT_IDS[a.dtype],
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
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
