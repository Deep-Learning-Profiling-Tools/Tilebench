import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

_DEFAULT_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 3}


def _tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    DIM = nargs["DIM"]
    nargs["desc_q"].block_shape = [1, 1, BLOCK_M, DIM]
    nargs["desc_k"].block_shape = [1, 1, BLOCK_N, DIM]
    nargs["desc_v"].block_shape = [1, 1, BLOCK_N, DIM]
    nargs["desc_o"].block_shape = [1, 1, BLOCK_M, DIM]


@triton.jit
def fwd_kernel(
    desc_q, desc_k, desc_v, sm_scale,
    desc_o,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)


    off_b = off_bs_head // HEAD
    off_h = off_bs_head % HEAD

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504


    q = desc_q.load([off_b, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, DIM])
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):


        k = desc_k.load([off_b, off_h, start_n, 0]).reshape([BLOCK_N, DIM])
        k = tl.trans(k)
        v = desc_v.load([off_b, off_h, start_n, 0]).reshape([BLOCK_N, DIM])

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new

    out_buffer = out_buffer / denom[:, None]
    desc_o.store(
        [off_b, off_h, start_m * BLOCK_M, 0],
        out_buffer.to(tl.float16).reshape([1, 1, BLOCK_M, DIM]),
    )


_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn},
            num_warps=nw, num_stages=ns,
            pre_hook=_tma_set_block_size_hook,
        )
        for bm in [64, 128]
        for bn in [32, 64, 128]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["SEQLEN", "DIM"],
)(fwd_kernel)


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, autotune: bool = False, **kwargs):


    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq ** 0.5)

    o = torch.empty_like(q)
    BS, HEAD, SEQLEN, _ = q.shape

    if autotune:

        dummy = [1, 1, 1, 1]
        desc_q = TensorDescriptor.from_tensor(q, dummy)
        desc_k = TensorDescriptor.from_tensor(k, dummy)
        desc_v = TensorDescriptor.from_tensor(v, dummy)
        desc_o = TensorDescriptor.from_tensor(o, dummy)
        grid = lambda meta: (triton.cdiv(SEQLEN, meta["BLOCK_M"]), BS * HEAD, 1)
        _fwd_kernel_autotuned[grid](
            desc_q, desc_k, desc_v, sm_scale,
            desc_o,
            BS, HEAD, SEQLEN,
            DIM=Lk,
            IS_CAUSAL=causal,
        )
    else:
        cfg = _DEFAULT_CONFIG
        BLOCK_M, BLOCK_N = cfg["BLOCK_M"], cfg["BLOCK_N"]
        desc_q = TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, Lk])
        desc_k = TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, Lk])
        desc_v = TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, Lk])
        desc_o = TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, Lk])
        grid = (triton.cdiv(SEQLEN, BLOCK_M), BS * HEAD, 1)
        fwd_kernel[grid](
            desc_q, desc_k, desc_v, sm_scale,
            desc_o,
            BS, HEAD, SEQLEN,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
            IS_CAUSAL=causal,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return o


def get_last_config() -> dict | None:
    cfg = getattr(_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
