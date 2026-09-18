```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    N_HEADS: tl.constexpr,
    N_CTX,
    HEAD_DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // N_HEADS
    off_h = pid_bh - off_b * N_HEADS

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    q_block = tl.make_block_ptr(
        base=q_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    k_block = tl.make_block_ptr(
        base=k_base,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )

    v_block = tl.make_block_ptr(
        base=v_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634

    if IS_CAUSAL:
        row_start = pid_m * BLOCK_M
        row_end = tl.minimum(N_CTX, (pid_m + 1) * BLOCK_M)

        # Tiles strictly before the causal boundary are fully valid and avoid
        # the per-element causal mask.  With long sequences this is the hot path.
        full_end = ((row_start + 1) // BLOCK_N) * BLOCK_N
        full_end = tl.minimum(full_end, N_CTX)

        for start_n in tl.range(0, full_end, BLOCK_N, num_stages=NUM_STAGES):
            k = tl.load(k_block)
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            v = tl.load(v_block)
            acc = tl.dot(p.to(tl.float16), v, acc)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))

        # Boundary/diagonal tile(s): causal mask required.
        for start_n in tl.range(full_end, row_end, BLOCK_N, num_stages=NUM_STAGES):
            k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            cols = start_n + offs_n
            valid = (cols[None, :] < N_CTX) & (offs_m[:, None] >= cols[None, :])
            qk = tl.where(valid, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(p.to(tl.float16), v, acc)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N, num_stages=NUM_STAGES):
            k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            cols = start_n + offs_n
            qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(p.to(tl.float16), v, acc)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    o_block = tl.make_block_ptr(
        base=o_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_os, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(o_block, acc.to(tl.float16), boundary_check=(0, 1))


def run(q, k, v, causal=True, **kwargs):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    n_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_D = 128
    num_warps = 8
    num_stages = 3

    sm_scale = head_dim ** -0.5
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_heads)

    _flash_attention_fwd_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        sm_scale,
        N_HEADS=n_heads,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(causal),
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "causal": bool(causal),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _flash_attention_fwd_kernel(
    q, k, v, output,
    qk_scale: float,
    SEQLEN: ConstInt,
    N_HEADS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    CAUSAL: ConstBool,
):
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)

    batch = bid_bh // N_HEADS
    head = bid_bh - batch * N_HEADS

    q_tile = ct.load(
        q,
        index=(batch, head, bid_m, 0),
        shape=(1, 1, BLOCK_M, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((BLOCK_M, BLOCK_D))

    m_i = ct.full((BLOCK_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((BLOCK_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=np.float32)

    offs_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n_base = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    if CAUSAL:
        full_tiles = ((bid_m * BLOCK_M + 1) // BLOCK_N)
        total_tiles = ct.cdiv((bid_m + 1) * BLOCK_M, BLOCK_N)

        # Hot path: tiles strictly below the causal boundary need no mask.
        for j in range(0, full_tiles):
            k_tile = ct.load(
                k,
                index=(batch, head, 0, j),
                shape=(1, 1, BLOCK_D, BLOCK_N),
                order=(0, 1, 3, 2),
            ).reshape((BLOCK_D, BLOCK_N))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32),
            )
            qk = qk * qk_scale

            m_new = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
            p = ct.exp2(qk - m_new, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
            acc = acc * alpha

            v_tile = ct.load(
                v,
                index=(batch, head, j, 0),
                shape=(1, 1, BLOCK_N, BLOCK_D),
            ).reshape((BLOCK_N, BLOCK_D))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            m_i = m_new

        # Boundary tile(s): apply causal and OOB masks.
        for j in range(full_tiles, total_tiles):
            k_tile = ct.load(
                k,
                index=(batch, head, 0, j),
                shape=(1, 1, BLOCK_D, BLOCK_N),
                order=(0, 1, 3, 2),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_D, BLOCK_N))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32),
            )
            qk = qk * qk_scale

            offs_n = j * BLOCK_N + offs_n_base
            valid = ct.broadcast_to(offs_n < SEQLEN, (BLOCK_M, BLOCK_N))
            valid = valid & (offs_m >= offs_n)
            qk = ct.where(valid, qk, -np.inf)

            m_new = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
            p = ct.exp2(qk - m_new, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
            acc = acc * alpha

            v_tile = ct.load(
                v,
                index=(batch, head, j, 0),
                shape=(1, 1, BLOCK_N, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_N, BLOCK_D))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            m_i = m_new
    else:
        n_tiles = ct.cdiv(SEQLEN, BLOCK_N)

        for j in range(0, n_tiles):
            k_tile = ct.load(
                k,
                index=(batch, head, 0, j),
                shape=(1, 1, BLOCK_D, BLOCK_N),
                order=(0, 1, 3, 2),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_D, BLOCK_N))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32),
            )
            qk = qk * qk_scale

            offs_n = j * BLOCK_N + offs_n_base
            valid = ct.broadcast_to(offs_n < SEQLEN, (BLOCK_M, BLOCK_N))
            qk = ct.where(valid, qk, -np.inf)

            m_new = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
            p = ct.exp2(qk - m_new, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_new, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
            acc = acc * alpha

            v_tile = ct.load(
                v,
                index=(batch, head, j, 0),
                shape=(1, 1, BLOCK_N, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_N, BLOCK_D))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            m_i = m_new

    out_tile = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    out_tile = ct.astype(out_tile, q.dtype).reshape((1, 1, BLOCK_M, BLOCK_D))

    ct.store(
        output,
        index=(batch, head, bid_m, 0),
        tile=out_tile,
    )


def run(q, k, v, causal=True, **kwargs):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    n_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_D = 128
    occupancy = 1

    sm_scale_log2 = (head_dim ** -0.5) * 1.4426950408889634

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(seq_len, BLOCK_M), batch_size * n_heads, 1)

    ct.launch(
        stream,
        grid,
        _flash_attention_fwd_kernel,
        (
            q, k, v, output,
            sm_scale_log2,
            seq_len,
            n_heads,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            bool(causal),
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_D": BLOCK_D,
        "occupancy": occupancy,
        "causal": bool(causal),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
