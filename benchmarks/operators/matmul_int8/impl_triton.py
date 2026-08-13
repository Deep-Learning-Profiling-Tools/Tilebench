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


_bt_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _b_transposed(b: torch.Tensor) -> torch.Tensor:
    bt = _bt_cache.get(b)
    if bt is None:
        bt = b.t().contiguous()
        _bt_cache[b] = bt
    return bt


def _tma_set_block_size_hook(nargs):
    bm = nargs["BLOCK_SIZE_M"]
    bn = nargs["BLOCK_SIZE_N"]
    bk = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [bm, bk]
    nargs["b_desc"].block_shape = [bn, bk]
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
    assert a.shape[1] == b.shape[0] * 4, (
        "Incompatible dims: A's K must equal 4 * B's K_b (B is packed 4-per-byte)"
    )
    assert a.is_contiguous(), "A must be contiguous"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    bt = _b_transposed(b)

    if autotune:

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
