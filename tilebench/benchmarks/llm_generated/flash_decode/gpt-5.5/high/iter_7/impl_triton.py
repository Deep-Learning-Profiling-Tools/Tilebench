import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_decode_single_kernel(
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
    BLOCK_D: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bh // HEADS
    h = pid_bh - b * HEADS

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Pass 1: global max LSE for this (batch, head).
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    for k in tl.range(0, NUM_BLOCKS, 1, num_stages=PIPE_STAGES):
        valid = k < valid_blocks
        cur_lse = tl.load(
            mid_o_lse + b * stride_lb + h * stride_lh + k * stride_lk,
            mask=valid,
            other=-float("inf"),
            cache_modifier=".ca",
            eviction_policy="evict_last",
        ).to(tl.float32)
        m_i = tl.maximum(m_i, cur_lse)

    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)

    # Pass 2: reduce numerator/denominator on the fixed global-max scale.
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)

    for k in tl.range(0, NUM_BLOCKS, 1, num_stages=PIPE_STAGES):
        valid = k < valid_blocks

        cur_lse = tl.load(
            mid_o_lse + b * stride_lb + h * stride_lh + k * stride_lk,
            mask=valid,
            other=-float("inf"),
            cache_modifier=".ca",
            eviction_policy="evict_last",
        ).to(tl.float32)

        beta = tl.where(
            valid,
            tl.exp2((cur_lse - m_safe) * 1.4426950408889634),
            0.0,
        )

        vals = tl.load(
            mid_o + b * stride_ob + h * stride_oh + k * stride_ok + offs_d * stride_od,
            mask=valid & mask_d,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)

        acc += vals * beta
        l_i += beta

    out = acc / (l_i + 1.0e-10)
    tl.store(output + pid_bh * HEAD_DIM + offs_d, out, mask=mask_d)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_D = 16
    PIPE_STAGES = 3
    num_warps = 1
    num_stages = 3

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_dchunks = triton.cdiv(head_dim, BLOCK_D)

    grid = (batch * heads, num_dchunks)
    _flash_decode_single_kernel[grid](
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
        BLOCK_D=BLOCK_D,
        PIPE_STAGES=PIPE_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_D": BLOCK_D,
            "PIPE_STAGES": PIPE_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "num_dchunks": num_dchunks,
            "num_blocks": num_blocks,
            "single_kernel": 1,
            "two_pass_global": 1,
            "exp2": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
