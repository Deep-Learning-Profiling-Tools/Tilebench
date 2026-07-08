"""Triton int8 GEMM with 2-bit packed B, host-side TMA descriptors.

A/B/C go through TensorDescriptors (tutorial-09 pattern, same as
batched_matmul / matmul_fp32_fp16_fp8 / streamk_matmul). This replaces
the cp.async (LDGSTS) load path with TMA (UTMALDG/UTMASTG in SASS).
Unlike TF32/fp16 tl.dot — which the TMA path flips to tcgen05 — int8
tl.dot still compiles to Hopper-era IMMA.16832 even when fed directly
from descriptors (probed with a plain int8 GEMM): this Triton version
has no tcgen05 lowering for integer dot. cuTile's UTCIMMA + TMA
remains a backend-level advantage on this op.

B is packed uint8 (K_b, N), 4 x 2-bit fields per byte; it is consumed
transposed to (N, K_b) so both operand boxes have the K axis innermost
(TMA swizzle fast path). The transposed copy is cached per input tensor
(WeakTensorKeyDictionary keyed by the caller's tensor — a .view()/temp
key would miss every call and re-pay the copy inside the measured
window) and the unmeasured warmup call pays for it. Unpacking stays in
registers: field i is (b >> 2i) & 3, minus 1, int8.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 4,
}

# B transposed to (N, K_b); cached by tensor identity, warmup pays the copy.
_bt_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _b_transposed(b: torch.Tensor) -> torch.Tensor:
    bt = _bt_cache.get(b)
    if bt is None:
        bt = b.t().contiguous()   # (N, K_b) row-major
        _bt_cache[b] = bt
    return bt


def _tma_set_block_size_hook(nargs):
    """Config pre_hook: rebind descriptor box shapes to this config's
    tile sizes before the autotuner times it."""
    bm = nargs["BLOCK_SIZE_M"]
    bn = nargs["BLOCK_SIZE_N"]
    bk = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [bm, bk]
    nargs["b_desc"].block_shape = [bn, bk]   # B passed transposed (N, K_b)
    nargs["c_desc"].block_shape = [bm, bn]


@triton.jit
def matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Int8 GEMM with 2-bit packed B (4 fields per byte).

    Outer loop walks K_b = K/4 in BLOCK_SIZE_K tiles, TMA-loading each
    packed B byte tile exactly once. The inner unrolled `for i in
    range(4)` loop unpacks the i-th 2-bit field in registers and pairs
    it with the A tile at K offset `i*K_b + j*BLOCK_SIZE_K`. Accumulator
    is int32 via tl.dot's integer MMA path.
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

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    one_i8 = tl.full((1,), 1, dtype=tl.int8)
    K_b: tl.constexpr = K // 4
    num_kb_tiles = tl.cdiv(K_b, BLOCK_SIZE_K)

    for j in range(0, num_kb_tiles):
        # TMA boxes zero-fill OOB lanes — no masks needed. b tile is
        # (BLOCK_N, BLOCK_K) from the transposed (N, K_b) view.
        b_uint8 = b_desc.load([offs_bn, j * BLOCK_SIZE_K])
        for i in range(4):
            k_pos = i * K_b + j * BLOCK_SIZE_K
            a = a_desc.load([offs_am, k_pos])
            mask_i = 3 << (2 * i)
            b = ((b_uint8 & mask_i) >> (2 * i)).to(tl.int8) - one_i8
            accumulator = tl.dot(a, b.T, accumulator, out_dtype=tl.int32)

    c_desc.store([offs_am, offs_bn], accumulator)


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
            pre_hook=_tma_set_block_size_hook,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [32, 64]
        for nw in [4, 8, 16]
        for ns in [3, 4]
        if bm * bn <= 128 * 256
        # int8 operand stages + int32 TMA store staging must fit B200's
        # 227 KB smem/block — the autotuner cannot skip OutOfResources.
        if (bm * bk + bn * bk) * ns + bm * bn * 4 <= 220_000
    ],
    key=["M", "N", "K"],
    warmup=3,
    rep=10,
)(matmul_kernel)


def _descriptors(a, bt, c, bm, bn, bk):
    a_desc = TensorDescriptor.from_tensor(a, [bm, bk])
    b_desc = TensorDescriptor.from_tensor(bt, [bn, bk])
    c_desc = TensorDescriptor.from_tensor(c, [bm, bn])
    return a_desc, b_desc, c_desc


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    """Triton int8 GEMM with 2-bit packed B."""
    assert a.shape[1] == b.shape[0] * 4, (
        "Incompatible dims: A's K must equal 4 * B's K_b (B is packed 4-per-byte)"
    )
    assert a.is_contiguous(), "A must be contiguous"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    bt = _b_transposed(b)   # (N, K_b); cached, warmup pays the copy

    if autotune:
        # Dummy box shape — the pre_hook overwrites it per config.
        a_desc, b_desc, c_desc = _descriptors(a, bt, c, 1, 1, 1)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        matmul_kernel_autotuned[grid](a_desc, b_desc, c_desc, M, N, K)
    else:
        cfg = _DEFAULT_CONFIG
        bm, bn, bk = (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"],
                      cfg["BLOCK_SIZE_K"])
        a_desc, b_desc, c_desc = _descriptors(a, bt, c, bm, bn, bk)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        matmul_kernel[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
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
