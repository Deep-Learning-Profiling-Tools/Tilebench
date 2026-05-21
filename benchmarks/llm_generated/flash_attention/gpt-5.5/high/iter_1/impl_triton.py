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
    scale_log2,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N

        for start_n in tl.range(0, full_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

        for start_n in tl.range(full_end, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    out = acc / l_i[:, None]

    tl.store(
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        out,
    )


_flash_attention_fwd_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["B", "H", "N_CTX", "D_HEAD", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N_CTX, D_HEAD = q.shape
    output = torch.empty_like(q)

    block_d = max(32, _next_power_of_2(int(D_HEAD)))
    scale_log2 = (1.0 / math.sqrt(float(D_HEAD))) * 1.4426950408889634
    causal = bool(causal)

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H)

    _flash_attention_fwd_autotuned[grid](
        q,
        k,
        v,
        output,
        scale_log2,
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
        B=B,
        H=H,
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_D=block_d,
        CAUSAL=causal,
    )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_autotuned, "best_config", None)
    if cfg is None:
        return None
    out = dict(cfg.kwargs)
    out["num_warps"] = cfg.num_warps
    out["num_stages"] = cfg.num_stages
    return out
