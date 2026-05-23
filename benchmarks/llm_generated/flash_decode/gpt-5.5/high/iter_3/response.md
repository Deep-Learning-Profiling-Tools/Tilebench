```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_decode_reduce_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    output,
    stride_ob,
    stride_oh,
    stride_ok,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lk,
    BLOCK_SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bh // HEADS
    h = pid_bh - b * HEADS

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_k = (offs_k < NUM_BLOCKS) & (offs_k < valid_blocks)
    mask_d = offs_d < HEAD_DIM

    lse = tl.load(
        mid_o_lse + b * stride_lb + h * stride_lh + offs_k * stride_lk,
        mask=offs_k < NUM_BLOCKS,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)
    lse = tl.where(mask_k, lse, -float("inf"))

    m = tl.max(lse, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)

    weights = tl.where(mask_k, tl.exp(lse - m_safe), 0.0)
    denom = tl.sum(weights, axis=0)

    vals = tl.load(
        mid_o
        + b * stride_ob
        + h * stride_oh
        + offs_k[:, None] * stride_ok
        + offs_d[None, :] * stride_od,
        mask=mask_k[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    acc = tl.sum(vals * weights[:, None], axis=0)
    out = acc / (denom + 1.0e-10)

    tl.store(output + pid_bh * HEAD_DIM + offs_d, out, mask=mask_d)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 512
    BLOCK_D = 16
    num_warps = 8
    num_stages = 4

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    grid = (batch * heads, triton.cdiv(head_dim, BLOCK_D))
    _flash_decode_reduce_kernel[grid](
        mid_o,
        mid_o_lse,
        b_seqlen,
        output,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_o_lse.stride(0),
        mid_o_lse.stride(1),
        mid_o_lse.stride(2),
        BLOCK_SEQ=block_seq_value,
        HEADS=heads,
        NUM_BLOCKS=num_blocks,
        HEAD_DIM=head_dim,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "BLOCK_D": BLOCK_D,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "one_pass": 1,
            "num_blocks": num_blocks,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _partial_flash_decode_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    tmp_acc,
    tmp_m,
    tmp_l,
    BLOCK_SEQ_VALUE: ConstInt,
    NUM_BLOCKS: ConstInt,
    HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_D: ConstInt,
):
    g = ct.bid(0)
    bh = ct.bid(1)
    dchunk = ct.bid(2)

    b = bh // HEADS
    h = bh - b * HEADS

    seq_len = ct.load(b_seqlen, index=(b,), shape=(), padding_mode=ct.PaddingMode.ZERO)
    valid_blocks = (seq_len + BLOCK_SEQ_VALUE - 1) // BLOCK_SEQ_VALUE

    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    m_i = ct.full((), -np.inf, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for kk in range(0, BLOCK_K):
        k = g * BLOCK_K + kk
        valid = k < valid_blocks

        cur_lse_raw = ct.astype(
            ct.load(
                mid_o_lse,
                index=(b, h, k),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )

        cur_m = ct.where(valid, cur_lse_raw, -np.inf)
        m_new = ct.maximum(m_i, cur_m)
        m_safe = ct.where(m_new == -np.inf, 0.0, m_new)

        alpha = ct.where(l_i > 0.0, ct.exp(m_i - m_safe), 0.0)
        beta = ct.where(valid, ct.exp(cur_m - m_safe), 0.0)

        vals = ct.astype(
            ct.load(
                mid_o,
                index=(b, h, k, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc * alpha + vals * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    ct.store(tmp_acc, index=(b, h, g, dchunk), tile=acc.reshape((1, 1, 1, BLOCK_D)))
    ct.store(tmp_m, index=(b, h, g), tile=m_i)
    ct.store(tmp_l, index=(b, h, g), tile=l_i)


@ct.kernel
def _final_flash_decode_kernel(
    tmp_acc,
    tmp_m,
    tmp_l,
    output,
    NUM_GROUPS: ConstInt,
    HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_G: ConstInt,
    BLOCK_D: ConstInt,
):
    bh = ct.bid(0)
    dchunk = ct.bid(1)

    b = bh // HEADS
    h = bh - b * HEADS

    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    m_i = ct.full((), -np.inf, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for gg in range(0, BLOCK_G):
        gm = ct.astype(
            ct.load(
                tmp_m,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )
        gl = ct.astype(
            ct.load(
                tmp_l,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ),
            np.float32,
        )

        m_new = ct.maximum(m_i, gm)
        m_safe = ct.where(m_new == -np.inf, 0.0, m_new)

        alpha = ct.where(l_i > 0.0, ct.exp(m_i - m_safe), 0.0)
        beta = ct.where(gl > 0.0, ct.exp(gm - m_safe), 0.0)

        group_acc = ct.astype(
            ct.load(
                tmp_acc,
                index=(b, h, gg, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc * alpha + group_acc * beta
        l_i = l_i * alpha + gl * beta
        m_i = m_new

    out = acc / (l_i + 1.0e-10)
    ct.store(output, index=(b, h, dchunk), tile=out.reshape((1, 1, BLOCK_D)))


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 64
    BLOCK_D = 64
    BLOCK_G = 8

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_groups = ct.cdiv(num_blocks, BLOCK_K)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    grid_partial = (num_groups, batch * heads, ct.cdiv(head_dim, BLOCK_D))
    ct.launch(
        stream,
        grid_partial,
        _partial_flash_decode_kernel,
        (
            mid_o,
            mid_o_lse,
            b_seqlen,
            tmp_acc,
            tmp_m,
            tmp_l,
            block_seq_value,
            num_blocks,
            heads,
            head_dim,
            BLOCK_K,
            BLOCK_D,
        ),
    )

    grid_final = (batch * heads, ct.cdiv(head_dim, BLOCK_D), 1)
    ct.launch(
        stream,
        grid_final,
        _final_flash_decode_kernel,
        (
            tmp_acc,
            tmp_m,
            tmp_l,
            output,
            num_groups,
            heads,
            head_dim,
            BLOCK_G,
            BLOCK_D,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "BLOCK_D": BLOCK_D,
            "BLOCK_G": BLOCK_G,
            "num_groups": num_groups,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
