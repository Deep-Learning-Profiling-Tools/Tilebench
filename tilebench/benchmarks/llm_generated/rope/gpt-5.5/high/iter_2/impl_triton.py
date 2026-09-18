import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _rope_kernel(q_ptr, cos_ptr, sin_ptr, out_ptr, seq_len,
                 N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_b = tl.program_id(2)

    half = HEAD_DIM // 2

    offs_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)

    mask_h = offs_h < N_HEADS
    mask_d = offs_d < half
    mask = mask_h[:, None] & mask_d[None, :]

    base = ((pid_b * seq_len + pid_s) * N_HEADS + offs_h[:, None]) * HEAD_DIM
    q1_offsets = base + offs_d[None, :]
    q2_offsets = q1_offsets + half

    trig_offsets = pid_s * half + offs_d

    q1 = tl.load(q_ptr + q1_offsets, mask=mask, other=0.0,
                 eviction_policy="evict_first")
    q2 = tl.load(q_ptr + q2_offsets, mask=mask, other=0.0,
                 eviction_policy="evict_first")

    cos_v = tl.load(cos_ptr + trig_offsets, mask=mask_d, other=0.0,
                    eviction_policy="evict_last")
    sin_v = tl.load(sin_ptr + trig_offsets, mask=mask_d, other=0.0,
                    eviction_policy="evict_last")

    # Mirror eager PyTorch half behavior: each multiply produces a tensor in
    # the output dtype, then add/sub produces the final output dtype.  Explicit
    # rtne downcasts avoid rare tie-rounding mismatches on fp16.
    p1 = (q1 * cos_v[None, :]).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    p2 = (q2 * sin_v[None, :]).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    out1 = (p1 - p2).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(out_ptr + q1_offsets, out1, mask=mask, eviction_policy="evict_first")

    p3 = (q2 * cos_v[None, :]).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    p4 = (q1 * sin_v[None, :]).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    out2 = (p3 + p4).to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(out_ptr + q2_offsets, out2, mask=mask, eviction_policy="evict_first")


def run(q, cos, sin):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    seq_len = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    half = head_dim // 2

    BLOCK_H = 32
    BLOCK_D = triton.next_power_of_2(half)
    num_warps = 8
    num_stages = 2

    grid = (seq_len, triton.cdiv(n_heads, BLOCK_H), batch_size)
    _rope_kernel[grid](
        q, cos, sin, output, seq_len,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
