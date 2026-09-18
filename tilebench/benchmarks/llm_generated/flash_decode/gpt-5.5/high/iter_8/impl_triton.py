import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _partial_flash_decode_vec_kernel(
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

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    offs_k = pid_g * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_k = (offs_k < NUM_BLOCKS) & (offs_k < valid_blocks)
    mask_d = offs_d < HEAD_DIM

    lse = tl.load(
        mid_o_lse + b * stride_lb + h * stride_lh + offs_k * stride_lk,
        mask=mask_k,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)

    m_i = tl.max(lse, axis=0)
    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)

    beta = tl.exp2((lse - m_safe) * 1.4426950408889634)
    beta = tl.where(mask_k, beta, 0.0)

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

    acc = tl.sum(vals * beta[:, None], axis=0)
    l_i = tl.sum(beta, axis=0)

    base = (pid_bh * NUM_GROUPS + pid_g) * HEAD_DIM
    tl.store(tmp_acc + base + offs_d, acc, mask=mask_d)
    tl.store(tmp_m + pid_bh * NUM_GROUPS + pid_g, m_i)
    tl.store(tmp_l + pid_bh * NUM_GROUPS + pid_g, l_i)


@triton.jit
def _final_flash_decode_vec_kernel(
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
    beta = tl.where((gl > 0.0) & mask_g, beta, 0.0)

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
    BLOCK_D = 128
    BLOCK_G = 16
    num_warps_partial = 4
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
    _partial_flash_decode_vec_kernel[grid_partial](
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
    _final_flash_decode_vec_kernel[grid_final](
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
            "num_dchunks": num_dchunks,
            "vectorized_reduce": 1,
            "exp2_merge": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
