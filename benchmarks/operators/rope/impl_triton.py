import torch
import triton
import triton.language as tl

next_power_of_2 = triton.next_power_of_2

_DEFAULT_CONFIG = {"ROPE_GROUP_SIZE": 16, "num_warps": 2, "num_stages": 2}


@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"]})
@triton.jit
def rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim        : tl.constexpr,
    n_heads         : tl.constexpr,
    BACKWARD_PASS   : tl.constexpr,
    BLOCK_SIZE      : tl.constexpr,
    ROPE_GROUP_SIZE : tl.constexpr,
):
    """RoPE embedding: Q * cos + rotate_half(Q) * sin (in-place on Q).

    Each CTA processes ROPE_GROUP_SIZE consecutive heads of one (batch, seq)
    row as one [ROPE_GROUP_SIZE, BLOCK_SIZE] 2D tile per half, sharing a
    single cos/sin row broadcast across the heads.
    """
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    col_mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + col_offsets,
                   mask=col_mask, other=0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + col_offsets,
                   mask=col_mask, other=0)

    if BACKWARD_PASS:
        sin1 = -sin1

    head_offsets = group_head_position * ROPE_GROUP_SIZE + tl.arange(0, ROPE_GROUP_SIZE)
    mask = (head_offsets[:, None] < n_heads) & col_mask[None, :]

    offs_q1 = row_position * Q_row_stride + head_offsets[:, None] * head_dim \
        + col_offsets[None, :]
    offs_q2 = offs_q1 + half_head_dim

    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

    tl.store(Q + offs_q1, Q1*cos1[None, :] - Q2*sin1[None, :], mask=mask)
    tl.store(Q + offs_q2, Q2*cos1[None, :] + Q1*sin1[None, :], mask=mask)


_rope_embedding_autotuned = triton.autotune(
    configs=[
        triton.Config({"ROPE_GROUP_SIZE": gs}, num_warps=nw, num_stages=ns)
        for gs in [1, 2, 4, 8, 16]
        for nw in [2, 4, 8]
        for ns in [2, 3]
    ],
    key=["seqlen", "head_dim"],
    restore_value=["Q"],
)(rope_embedding)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        block_size: int = None, autotune: bool = False):
    # RoPE is in-place; clone so the caller's q stays pristine across backends.
    output = q.clone().contiguous()
    batch, seq_len, n_heads, head_dim = output.shape

    BLOCK_SIZE = next_power_of_2(head_dim // 2)
    n_rows = batch * seq_len

    if autotune:
        grid = lambda meta: (n_rows, triton.cdiv(n_heads, meta["ROPE_GROUP_SIZE"]))
        _rope_embedding_autotuned[grid](
            output,   output.stride(1),
            cos,      cos.stride(0),
            sin,      sin.stride(0),
            seq_len,
            head_dim,
            n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        cfg = _DEFAULT_CONFIG
        n_groups = triton.cdiv(n_heads, cfg["ROPE_GROUP_SIZE"])
        rope_embedding[(n_rows, n_groups)](
            output,   output.stride(1),
            cos,      cos.stride(0),
            sin,      sin.stride(0),
            seq_len,
            head_dim,
            n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            ROPE_GROUP_SIZE=cfg["ROPE_GROUP_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_rope_embedding_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "ROPE_GROUP_SIZE": cfg.kwargs["ROPE_GROUP_SIZE"],
        "num_warps":       cfg.num_warps,
        "num_stages":      cfg.num_stages,
    }
