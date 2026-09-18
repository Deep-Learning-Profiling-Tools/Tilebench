import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _partial_flash_decode_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    block_seq_tensor,
    tmp_acc,
    tmp_m,
    tmp_l,
    block_seq_scalar,
    heads,
    num_blocks,
    head_dim,
    num_groups,
    stride_ob,
    stride_oh,
    stride_ok,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lk,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SEQ_IS_TENSOR: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_bh // heads
    h = pid_bh - b * heads

    if BLOCK_SEQ_IS_TENSOR:
        block_seq = tl.load(block_seq_tensor + 0)
    else:
        block_seq = block_seq_scalar

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + block_seq - 1) // block_seq

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)

    for kk in tl.range(0, BLOCK_K, 1, num_stages=PIPE_STAGES):
        k = pid_g * BLOCK_K + kk
        valid = (k < num_blocks) & (k < valid_blocks)

        cur_lse = tl.load(
            mid_o_lse + b * stride_lb + h * stride_lh + k * stride_lk,
            mask=valid,
            other=-float("inf"),
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)

        cur_m = tl.where(valid, cur_lse, -float("inf"))
        m_new = tl.maximum(m_i, cur_m)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)

        alpha = tl.where(l_i > 0.0, tl.exp(m_i - m_safe), 0.0)
        beta = tl.where(valid, tl.exp(cur_m - m_safe), 0.0)

        vals = tl.load(
            mid_o + b * stride_ob + h * stride_oh + k * stride_ok + offs_d * stride_od,
            mask=valid & mask_d,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)

        acc = acc * alpha + vals * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    base = (pid_bh * num_groups + pid_g) * head_dim
    tl.store(tmp_acc + base + offs_d, acc, mask=mask_d)
    tl.store(tmp_m + pid_bh * num_groups + pid_g, m_i)
    tl.store(tmp_l + pid_bh * num_groups + pid_g, l_i)


@triton.jit
def _final_flash_decode_kernel(
    tmp_acc,
    tmp_m,
    tmp_l,
    output,
    head_dim,
    num_groups,
    BLOCK_D: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)

    for gg in tl.range(0, num_groups, 1, num_stages=PIPE_STAGES):
        gm = tl.load(
            tmp_m + pid_bh * num_groups + gg,
            other=-float("inf"),
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        gl = tl.load(
            tmp_l + pid_bh * num_groups + gg,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)

        m_new = tl.maximum(m_i, gm)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)

        alpha = tl.where(l_i > 0.0, tl.exp(m_i - m_safe), 0.0)
        beta = tl.where(gl > 0.0, tl.exp(gm - m_safe), 0.0)

        group_acc = tl.load(
            tmp_acc + (pid_bh * num_groups + gg) * head_dim + offs_d,
            mask=mask_d,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)

        acc = acc * alpha + group_acc * beta
        l_i = l_i * alpha + gl * beta
        m_i = m_new

    out = acc / (l_i + 1.0e-10)
    tl.store(output + pid_bh * head_dim + offs_d, out, mask=mask_d)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 64
    PIPE_STAGES = 3
    num_warps = 2
    num_stages = 3

    num_groups = triton.cdiv(num_blocks, BLOCK_K)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    block_seq_is_tensor = isinstance(block_seq, torch.Tensor)
    if block_seq_is_tensor:
        block_seq_tensor = block_seq
        block_seq_scalar = 1
    else:
        block_seq_tensor = b_seqlen
        block_seq_scalar = int(block_seq)

    grid_partial = (num_groups, batch * heads, triton.cdiv(head_dim, BLOCK_D))
    _partial_flash_decode_kernel[grid_partial](
        mid_o,
        mid_o_lse,
        b_seqlen,
        block_seq_tensor,
        tmp_acc,
        tmp_m,
        tmp_l,
        block_seq_scalar,
        heads,
        num_blocks,
        head_dim,
        num_groups,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_o_lse.stride(0),
        mid_o_lse.stride(1),
        mid_o_lse.stride(2),
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        BLOCK_SEQ_IS_TENSOR=block_seq_is_tensor,
        PIPE_STAGES=PIPE_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    grid_final = (batch * heads, triton.cdiv(head_dim, BLOCK_D))
    _final_flash_decode_kernel[grid_final](
        tmp_acc,
        tmp_m,
        tmp_l,
        output,
        head_dim,
        num_groups,
        BLOCK_D=BLOCK_D,
        PIPE_STAGES=PIPE_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "BLOCK_D": BLOCK_D,
            "PIPE_STAGES": PIPE_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "num_groups": num_groups,
            "block_seq_is_tensor": block_seq_is_tensor,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
