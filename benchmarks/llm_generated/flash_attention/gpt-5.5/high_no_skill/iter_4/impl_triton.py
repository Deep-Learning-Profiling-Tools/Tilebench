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
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    FULL_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_vd = pid_d * BLOCK_VD + tl.arange(0, BLOCK_VD)

    base = pid_bh * N_CTX * HEAD_DIM

    q = tl.load(
        q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    qk_scale = SM_SCALE * 1.4426950408889634

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, BLOCK_VD), 0.0, tl.float32)

    row_start = pid_m * BLOCK_M

    if CAUSAL and not FULL_CAUSAL:
        off_end = (row_start // BLOCK_N) * BLOCK_N

        for start_n in tl.range(0, off_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cols = start_n + offs_n

            k_t = tl.load(
                k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
                mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
                other=0.0,
            )

            qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale
            qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            p = tl.exp2(qk - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)

            v = tl.load(
                v_ptr + base + cols[:, None] * HEAD_DIM + offs_vd[None, :],
                mask=(cols[:, None] < N_CTX) & (offs_vd[None, :] < HEAD_DIM),
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        end_n = tl.minimum(row_start + BLOCK_M, N_CTX)
        for start_n in tl.range(off_end, end_n, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cols = start_n + offs_n

            k_t = tl.load(
                k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
                mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
                other=0.0,
            )

            qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale
            qk = tl.where(
                (cols[None, :] <= offs_m[:, None]) & (cols[None, :] < N_CTX),
                qk,
                -float("inf"),
            )

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            p = tl.exp2(qk - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)

            v = tl.load(
                v_ptr + base + cols[:, None] * HEAD_DIM + offs_vd[None, :],
                mask=(cols[:, None] < N_CTX) & (offs_vd[None, :] < HEAD_DIM),
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            cols = start_n + offs_n

            k_t = tl.load(
                k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
                mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
                other=0.0,
            )

            qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale

            if CAUSAL:
                qk = tl.where(
                    (cols[None, :] <= offs_m[:, None]) & (cols[None, :] < N_CTX),
                    qk,
                    -float("inf"),
                )
            else:
                qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            p = tl.exp2(qk - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)

            v = tl.load(
                v_ptr + base + cols[:, None] * HEAD_DIM + offs_vd[None, :],
                mask=(cols[:, None] < N_CTX) & (offs_vd[None, :] < HEAD_DIM),
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = acc / l_i[:, None]

    tl.store(
        o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_vd[None, :],
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_vd[None, :] < HEAD_DIM),
    )


@triton.jit
def _flash_attention_early_fix_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    EARLY_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    rows = tl.arange(0, EARLY_M)
    cols = tl.arange(0, EARLY_M)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * N_CTX * HEAD_DIM

    q = tl.load(
        q_ptr + base + rows[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(rows[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_t = tl.load(
        k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
        mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
        other=0.0,
    )

    scores = tl.dot(q, k_t, out_dtype=tl.float32) * SM_SCALE
    scores = tl.where(
        (cols[None, :] <= rows[:, None]) & (cols[None, :] < N_CTX),
        scores,
        -float("inf"),
    )

    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    l = tl.sum(p, axis=1)

    acc = tl.full((EARLY_M, BLOCK_D), 0.0, tl.float32)

    for j in tl.static_range(0, EARLY_M):
        vj = tl.load(
            v_ptr + base + j * HEAD_DIM + offs_d,
            mask=(j < N_CTX) & (offs_d < HEAD_DIM),
            other=0.0,
        )
        pj = p[:, j] / l
        acc += pj[:, None] * vj[None, :]

    tl.store(
        o_ptr + base + rows[:, None] * HEAD_DIM + offs_d[None, :],
        acc,
        mask=(rows[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


_flash_attention_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_VD": 64, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_VD": 64, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_VD": 64, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_VD": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_VD": 128, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_VD": 128, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
    ],
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, n_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)

    block_d = _next_power_of_2(head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)
    batch_heads = batch_size * n_heads

    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_M"]),
        batch_heads,
        triton.cdiv(head_dim, meta["BLOCK_VD"]),
    )

    _flash_attention_fwd_kernel_autotuned[grid](
        q,
        k,
        v,
        output,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
        CAUSAL=bool(causal),
        BLOCK_D=block_d,
    )

    if bool(causal):
        _flash_attention_early_fix_kernel[(batch_heads,)](
            q,
            k,
            v,
            output,
            N_CTX=seq_len,
            HEAD_DIM=head_dim,
            SM_SCALE=sm_scale,
            EARLY_M=16,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "BLOCK_VD": cfg.kwargs["BLOCK_VD"],
        "FULL_CAUSAL": cfg.kwargs["FULL_CAUSAL"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
