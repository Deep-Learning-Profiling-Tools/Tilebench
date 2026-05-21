import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    N_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // N_HEADS
    h = pid_bh - b * N_HEADS

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    q = tl.load(
        q_ptr
        + q_base
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        loop_end = tl.minimum(start_m + BLOCK_M, N_CTX)
    else:
        loop_end = N_CTX

    for start_n in tl.range(0, loop_end, BLOCK_N):
        offs_n = start_n + offs_n_base

        k = tl.load(
            k_ptr
            + k_base
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_kn,
            mask=(offs_d[:, None] < HEAD_DIM) & (offs_n[None, :] < N_CTX),
            other=0.0,
        )

        qk = tl.dot(q, k, out_dtype=tl.float32) * SM_SCALE_LOG2
        qk = tl.where(offs_n[None, :] < N_CTX, qk, -float("inf"))

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)

        v = tl.load(
            v_ptr
            + v_base
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(
        o_ptr
        + o_base
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


_flash_attention_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    batch_size, n_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)

    block_d = max(32, _next_power_of_2(int(head_dim)))
    sm_scale_log2 = (1.0 / math.sqrt(float(head_dim))) * 1.4426950408889634

    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch_size * n_heads)

    _flash_attention_fwd_kernel_autotuned[grid](
        q,
        k,
        v,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        N_HEADS=n_heads,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        SM_SCALE_LOG2=sm_scale_log2,
        CAUSAL=bool(causal),
    )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
