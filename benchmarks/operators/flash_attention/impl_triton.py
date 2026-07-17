import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 8, "num_stages": 4}


@triton.jit
def fwd_kernel(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom))
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16))


_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn},
            num_warps=nw, num_stages=ns,
        )
        for bm in [64, 128]
        for bn in [32, 64, 128]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["SEQLEN", "DIM"],
)(fwd_kernel)


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, autotune: bool = False, **kwargs):

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq ** 0.5)

    o = torch.empty_like(q)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    if autotune:
        grid = lambda meta: (triton.cdiv(q.shape[2], meta["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        _fwd_kernel_autotuned[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            DIM=Lk,
            IS_CAUSAL=causal,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(q.shape[2], cfg["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], DIM=Lk,
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
