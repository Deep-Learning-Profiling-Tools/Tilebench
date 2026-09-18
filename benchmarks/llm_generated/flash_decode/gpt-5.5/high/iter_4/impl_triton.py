import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_decode_lock_merge_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    output,
    tmp_acc,
    tmp_m,
    tmp_l,
    sync,
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
    NUM_STATES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_bh // HEADS
    h = pid_bh - b * HEADS

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    local_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    seq_len = tl.load(b_seqlen + b)
    valid_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)

    for kk in tl.range(0, BLOCK_K, 1, num_stages=PIPE_STAGES):
        k = pid_g * BLOCK_K + kk
        valid = (k < NUM_BLOCKS) & (k < valid_blocks)

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

    state_id = pid_bh * tl.cdiv(HEAD_DIM, BLOCK_D) + pid_d
    lock_ptr = sync + state_id
    count_ptr = sync + NUM_STATES + state_id

    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") == 1:
        pass

    count = tl.load(count_ptr)
    has_state = count > 0

    state_m = tl.load(tmp_m + state_id, mask=has_state, other=-float("inf")).to(tl.float32)
    state_l = tl.load(tmp_l + state_id, mask=has_state, other=0.0).to(tl.float32)
    state_acc = tl.load(
        tmp_acc + state_id * BLOCK_D + local_d,
        mask=has_state,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    merge_m = tl.maximum(state_m, m_i)
    merge_safe = tl.where(merge_m == -float("inf"), 0.0, merge_m)

    state_scale = tl.where(state_l > 0.0, tl.exp(state_m - merge_safe), 0.0)
    group_scale = tl.where(l_i > 0.0, tl.exp(m_i - merge_safe), 0.0)

    merged_acc = state_acc * state_scale + acc * group_scale
    merged_l = state_l * state_scale + l_i * group_scale

    tl.store(tmp_acc + state_id * BLOCK_D + local_d, merged_acc)
    tl.store(tmp_m + state_id, merge_m)
    tl.store(tmp_l + state_id, merged_l)

    new_count = count + 1
    tl.store(count_ptr, new_count)

    if new_count == NUM_GROUPS:
        out = merged_acc / (merged_l + 1.0e-10)
        tl.store(output + pid_bh * HEAD_DIM + offs_d, out, mask=mask_d)

    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 64
    PIPE_STAGES = 3
    num_warps = 2
    num_stages = 3

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_groups = triton.cdiv(num_blocks, BLOCK_K)
    num_dchunks = triton.cdiv(head_dim, BLOCK_D)
    num_states = batch * heads * num_dchunks

    tmp_acc = torch.empty((num_states, BLOCK_D), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((num_states,), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((num_states,), device=mid_o.device, dtype=torch.float32)
    sync = torch.zeros((2 * num_states,), device=mid_o.device, dtype=torch.int32)

    grid = (num_groups, batch * heads, num_dchunks)
    _flash_decode_lock_merge_kernel[grid](
        mid_o,
        mid_o_lse,
        b_seqlen,
        output,
        tmp_acc,
        tmp_m,
        tmp_l,
        sync,
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
        NUM_STATES=num_states,
        BLOCK_K=BLOCK_K,
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
            "num_states": num_states,
            "spin_lock_merge": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
