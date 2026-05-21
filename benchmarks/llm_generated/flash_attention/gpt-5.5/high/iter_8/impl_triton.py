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
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    FULL_CAUSAL: tl.constexpr,
):
    pid_m_raw = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Serpentine scheduling across heads: odd heads run query blocks in reverse.
    # This preserves per-head K/V locality while reducing the long causal tail.
    n_m_blocks = tl.cdiv(N_CTX, BLOCK_M)
    pid_m = tl.where((pid_bh % 2) == 1, n_m_blocks - 1 - pid_m_raw, pid_m_raw)

    base = pid_bh * N_CTX * D_HEAD

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + base
        + offs_m[:, None] * D_HEAD
        + offs_d[None, :]
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        if FULL_CAUSAL:
            # Regular full sweep variant. Autotune can select this only when
            # avoiding block-level causal branches beats skipping future tiles.
            for start_n in tl.range(0, N_CTX, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    k_ptr
                    + base
                    + offs_d[:, None]
                    + offs_n[None, :] * D_HEAD,
                    eviction_policy="evict_last",
                )

                qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
                qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                alpha = tl.exp2(m_i - m_ij)

                v = tl.load(
                    v_ptr
                    + base
                    + offs_n[:, None] * D_HEAD
                    + offs_d[None, :],
                    eviction_policy="evict_last",
                )

                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij
        else:
            full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N

            for start_n in tl.range(0, full_end, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    k_ptr
                    + base
                    + offs_d[:, None]
                    + offs_n[None, :] * D_HEAD,
                    eviction_policy="evict_last",
                )

                qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                alpha = tl.exp2(m_i - m_ij)

                v = tl.load(
                    v_ptr
                    + base
                    + offs_n[:, None] * D_HEAD
                    + offs_d[None, :],
                    eviction_policy="evict_last",
                )

                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij

            hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

            for start_n in tl.range(full_end, hi, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    k_ptr
                    + base
                    + offs_d[:, None]
                    + offs_n[None, :] * D_HEAD,
                    eviction_policy="evict_last",
                )

                qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
                qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                alpha = tl.exp2(m_i - m_ij)

                v = tl.load(
                    v_ptr
                    + base
                    + offs_n[:, None] * D_HEAD
                    + offs_d[None, :],
                    eviction_policy="evict_last",
                )

                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + base
                + offs_d[:, None]
                + offs_n[None, :] * D_HEAD,
                eviction_policy="evict_last",
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + base
                + offs_n[:, None] * D_HEAD
                + offs_d[None, :],
                eviction_policy="evict_last",
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    out = acc / l_i[:, None]

    tl.store(
        o_ptr
        + base
        + offs_m[:, None] * D_HEAD
        + offs_d[None, :],
        out,
    )


_flash_attention_fwd_autotuned = triton.autotune(
    configs=[
        # Verified skip-future causal variants from the best clean iteration.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        # A few safe additional pipeline/warp points.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "FULL_CAUSAL": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "FULL_CAUSAL": False}, num_warps=4, num_stages=4),
        # Branch-free full-sweep alternatives for cases where regularity wins.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "FULL_CAUSAL": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "FULL_CAUSAL": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "FULL_CAUSAL": True}, num_warps=4, num_stages=3),
    ],
    key=["N_CTX", "D_HEAD", "CAUSAL"],
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
