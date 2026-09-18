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
