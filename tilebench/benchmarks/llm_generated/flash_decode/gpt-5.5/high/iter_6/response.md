```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _partial_flash_decode_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    tmp_acc,
    tmp_m,
    tmp_l,
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
    NUM_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_bh // HEADS
    h = pid_bh - b * HEADS

    offs_k = pid_g * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    valid_k = (offs_k < NUM_BLOCKS) & (offs_k < valid_blocks)

    lse = tl.load(
        mid_o_lse + b * stride_lb + h * stride_lh + offs_k * stride_lk,
        mask=valid_k,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    m_i = tl.max(lse, axis=0)
    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)

    beta = tl.exp2((lse - m_safe) * 1.4426950408889634)
    beta = tl.where(valid_k, beta, 0.0)

    vals = tl.load(
        mid_o
        + b * stride_ob
        + h * stride_oh
        + offs_k[:, None] * stride_ok
        + offs_d[None, :] * stride_od,
        mask=valid_k[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    acc = tl.sum(vals * beta[:, None], axis=0)
    l_i = tl.sum(beta, axis=0)

    base = (pid_bh * NUM_GROUPS + pid_g) * HEAD_DIM
    tl.store(tmp_acc + base + offs_d, acc, mask=mask_d)

    # m/l are independent of dchunk.  Store them once to avoid duplicate writes.
    tl.store(tmp_m + pid_bh * NUM_GROUPS + pid_g, m_i, mask=pid_d == 0)
    tl.store(tmp_l + pid_bh * NUM_GROUPS + pid_g, l_i, mask=pid_d == 0)


@triton.jit
def _final_flash_decode_kernel(
    tmp_acc,
    tmp_m,
    tmp_l,
    output,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_G)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_g = offs_g < NUM_GROUPS
    mask_d = offs_d < HEAD_DIM

    gm = tl.load(
        tmp_m + pid_bh * NUM_GROUPS + offs_g,
        mask=mask_g,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    gl = tl.load(
        tmp_l + pid_bh * NUM_GROUPS + offs_g,
        mask=mask_g,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    m_i = tl.max(gm, axis=0)
    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)

    beta = tl.exp2((gm - m_safe) * 1.4426950408889634)
    beta = tl.where(gl > 0.0, beta, 0.0)

    group_acc = tl.load(
        tmp_acc + (pid_bh * NUM_GROUPS + offs_g[:, None]) * HEAD_DIM + offs_d[None, :],
        mask=mask_g[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    acc = tl.sum(group_acc * beta[:, None], axis=0)
    l_i = tl.sum(gl * beta, axis=0)

    out = acc / (l_i + 1.0e-10)
    tl.store(output + pid_bh * HEAD_DIM + offs_d, out, mask=mask_d)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 64
    BLOCK_G = 16
    num_warps_partial = 8
    num_warps_final = 4
    num_stages = 3

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_groups = triton.cdiv(num_blocks, BLOCK_K)
    num_dchunks = triton.cdiv(head_dim, BLOCK_D)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    grid_partial = (num_groups, batch * heads, num_dchunks)
    _partial_flash_decode_kernel[grid_partial](
        mid_o,
        mid_o_lse,
        b_seqlen,
        tmp_acc,
        tmp_m,
        tmp_l,
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
        NUM_GROUPS=num_groups,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps_partial,
        num_stages=num_stages,
    )

    grid_final = (batch * heads, num_dchunks)
    _final_flash_decode_kernel[grid_final](
        tmp_acc,
        tmp_m,
        tmp_l,
        output,
        HEAD_DIM=head_dim,
        NUM_GROUPS=num_groups,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps_final,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "BLOCK_D": BLOCK_D,
            "BLOCK_G": BLOCK_G,
            "num_warps_partial": num_warps_partial,
            "num_warps_final": num_warps_final,
            "num_stages": num_stages,
            "num_groups": num_groups,
            "vectorized_reduce": 1,
            "exp2_merge": 1,
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

    offs_k = g * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)
    valid = (offs_k < NUM_BLOCKS) & (offs_k < valid_blocks)

    lse_raw = ct.astype(
        ct.load(
            mid_o_lse,
            index=(b, h, g),
            shape=(1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
        ).reshape((BLOCK_K,)),
        np.float32,
    )

    lse = ct.where(valid, lse_raw, -np.inf)
    m_i = ct.max(lse)
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)

    beta = ct.exp2((lse - m_safe) * 1.4426950408889634, flush_to_zero=True)
    beta = ct.where(valid, beta, 0.0)

    vals = ct.astype(
        ct.load(
            mid_o,
            index=(b, h, g, dchunk),
            shape=(1, 1, BLOCK_K, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_K, BLOCK_D)),
        np.float32,
    )

    acc = ct.sum(vals * beta[:, None], axis=0)
    l_i = ct.sum(beta)

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

    gm = ct.astype(
        ct.load(
            tmp_m,
            index=(b, h, 0),
            shape=(1, 1, BLOCK_G),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
        ).reshape((BLOCK_G,)),
        np.float32,
    )

    gl = ct.astype(
        ct.load(
            tmp_l,
            index=(b, h, 0),
            shape=(1, 1, BLOCK_G),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_G,)),
        np.float32,
    )

    m_i = ct.max(gm)
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)

    beta = ct.exp2((gm - m_safe) * 1.4426950408889634, flush_to_zero=True)
    beta = ct.where(gl > 0.0, beta, 0.0)

    group_acc = ct.astype(
        ct.load(
            tmp_acc,
            index=(b, h, 0, dchunk),
            shape=(1, 1, BLOCK_G, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_G, BLOCK_D)),
        np.float32,
    )

    acc = ct.sum(group_acc * beta[:, None], axis=0)
    l_i = ct.sum(gl * beta)

    out = acc / (l_i + 1.0e-10)
    ct.store(output, index=(b, h, dchunk), tile=out.reshape((1, 1, BLOCK_D)))


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 128
    BLOCK_G = 16
    occupancy_partial = 2
    occupancy_final = 4

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_groups = ct.cdiv(num_blocks, BLOCK_K)
    num_dchunks = ct.cdiv(head_dim, BLOCK_D)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    partial_kernel = _partial_flash_decode_kernel.with_hints(occupancy=occupancy_partial)
    final_kernel = _final_flash_decode_kernel.with_hints(occupancy=occupancy_final)

    grid_partial = (num_groups, batch * heads, num_dchunks)
    ct.launch(
        stream,
        grid_partial,
        partial_kernel,
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

    grid_final = (batch * heads, num_dchunks, 1)
    ct.launch(
        stream,
        grid_final,
        final_kernel,
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
            "occupancy_partial": occupancy_partial,
            "occupancy_final": occupancy_final,
            "vectorized_reduce": 1,
            "exp2_merge": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
