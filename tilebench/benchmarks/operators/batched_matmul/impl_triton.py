import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilebench.hardware import detect_arch

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUPSIZE": 8,
    "num_warps": 4,
    "num_stages": 4,
}

# Hardware-compatibility fallback, not AMD performance tuning. On CDNA3
# (MI300X) the default above needs 98304 B of LDS for fp32, over gfx942's
# 65536 B, so the non-autotuned path cannot launch. This keeps its 128x128x32
# tile, GROUPSIZE and num_warps and only lowers the pipeline depth from 4 to 3
# stages, the first depth that fits (fp32: 65536 B). It is a candidate of the
# autotune space below, which is unchanged on every architecture.
_CDNA3_DEFAULT_CONFIG = {**_DEFAULT_CONFIG, "num_stages": 3}
_BUILTIN_DEFAULT_CONFIG = _DEFAULT_CONFIG


def _default_config() -> dict:
    """Fixed config of the non-autotuned path, by the GPU actually present.

    A config written into _DEFAULT_CONFIG (a profiler harness replaying an
    autotune winner) is used as given: the CDNA3 fallback only replaces the
    builtin default."""
    if _DEFAULT_CONFIG is _BUILTIN_DEFAULT_CONFIG and detect_arch() == "cdna3":
        return _CDNA3_DEFAULT_CONFIG
    return _DEFAULT_CONFIG


_DT_IDS = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


_bt_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _b_transposed(B_flat: torch.Tensor, BATCH: int, K: int, N: int) -> torch.Tensor:


    bt = _bt_cache.get(B_flat)
    if bt is None:
        bt = B_flat.view(BATCH, K, N).transpose(1, 2).contiguous()
        _bt_cache[B_flat] = bt
    return bt


def _tma_set_block_size_hook(nargs):
    bm = nargs["BLOCK_SIZE_M"]
    bn = nargs["BLOCK_SIZE_N"]
    bk = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [1, bm, bk]
    nargs["b_desc"].block_shape = [1, bn, bk]
    nargs["c_desc"].block_shape = [1, bm, bn]


@triton.jit
def bmm_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    DT_ID: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUPSIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    bid = tl.program_id(2)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_m, pid_n = tl.swizzle2d(pid0, pid1, num_pid_m, num_pid_n, GROUPSIZE)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(tl.cdiv(K, BLOCK_SIZE_K)):

        a = a_desc.load([bid, offs_am, k * BLOCK_SIZE_K])
        b = b_desc.load([bid, offs_bn, k * BLOCK_SIZE_K])
        a2 = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        b2 = tl.reshape(b, (BLOCK_SIZE_N, BLOCK_SIZE_K))


        acc = tl.dot(a2, b2.T, acc, input_precision="tf32")

    c = tl.reshape(acc.to(c_desc.dtype), (1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([bid, offs_am, offs_bn], c)


_bmm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
             "GROUPSIZE": gs},
            num_warps=nw,
            num_stages=ns,
            pre_hook=_tma_set_block_size_hook,
        )
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for bk in [32, 64]
        for gs in [1, 8]
        for nw in [4, 8]
        for ns in [2, 3, 4]
        if (bm * bk + bn * bk) * 4 * ns + bm * bn * 4 <= 220_000
    ],
    key=["M", "N", "K", "DT_ID"],
    warmup=1,
    rep=3,
)(bmm_kernel)


def _descriptors(a_3d, bt_3d, c_3d, bm, bn, bk):
    a_desc = TensorDescriptor.from_tensor(a_3d, [1, bm, bk])
    b_desc = TensorDescriptor.from_tensor(bt_3d, [1, bn, bk])
    c_desc = TensorDescriptor.from_tensor(c_3d, [1, bm, bn])
    return a_desc, b_desc, c_desc


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    a_3d = A.view(BATCH, M, K)
    c = torch.empty(BATCH * M * N, dtype=A.dtype, device=A.device)
    c_3d = c.view(BATCH, M, N)
    bt_3d = _b_transposed(B, BATCH, K, N)
    dt_id = _DT_IDS[A.dtype]

    if autotune:

        a_desc, b_desc, c_desc = _descriptors(a_3d, bt_3d, c_3d, 1, 1, 1)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            BATCH,
        )
        _bmm_kernel_autotuned[grid](a_desc, b_desc, c_desc, M, N, K,
                                    DT_ID=dt_id)
    else:
        cfg = _default_config()
        bm, bn, bk = (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"],
                      cfg["BLOCK_SIZE_K"])
        a_desc, b_desc, c_desc = _descriptors(a_3d, bt_3d, c_3d, bm, bn, bk)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn), BATCH)
        bmm_kernel[grid](
            a_desc, b_desc, c_desc, M, N, K,
            DT_ID=dt_id,
            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
            GROUPSIZE=cfg["GROUPSIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return c


def get_last_config() -> dict | None:
    cfg = getattr(_bmm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_M": cfg.kwargs["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": cfg.kwargs["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": cfg.kwargs["BLOCK_SIZE_K"],
        "GROUPSIZE": cfg.kwargs["GROUPSIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
