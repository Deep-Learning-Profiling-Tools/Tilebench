```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        asm="add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        asm="sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_kernel(q_ptr, cos_ptr, sin_ptr, out_ptr, seq_len,
                 N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
                 OUT_FP32: tl.constexpr):
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

    q1_f = q1.to(tl.float32)
    q2_f = q2.to(tl.float32)
    cos_f = cos_v.to(tl.float32)
    sin_f = sin_v.to(tl.float32)

    cos_b = tl.broadcast_to(cos_f[None, :], (BLOCK_H, BLOCK_D))
    sin_b = tl.broadcast_to(sin_f[None, :], (BLOCK_H, BLOCK_D))

    if OUT_FP32:
        # PyTorch eager materializes each multiply before the following add/sub.
        # Opaque IEEE-rounding asm prevents contraction into FFMA and matches the
        # separate TensorIterator kernels' fp32 rounding points.
        p1 = _mul_rn_f32(q1_f, cos_b)
        p2 = _mul_rn_f32(q2_f, sin_b)
        out1 = _sub_rn_f32(p1, p2)
        tl.store(out_ptr + q1_offsets, out1, mask=mask,
                 eviction_policy="evict_first")

        p3 = _mul_rn_f32(q2_f, cos_b)
        p4 = _mul_rn_f32(q1_f, sin_b)
        out2 = _add_rn_f32(p3, p4)
        tl.store(out_ptr + q2_offsets, out2, mask=mask,
                 eviction_policy="evict_first")
    else:
        # For pure fp16 inputs PyTorch creates fp16 product temporaries, then the
        # final add/sub writes fp16. Keep the same rounding points explicitly.
        p1_h = _mul_rn_f32(q1_f, cos_b).to(tl.float16)
        p2_h = _mul_rn_f32(q2_f, sin_b).to(tl.float16)
        p1_h = p1_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        p2_h = p2_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        out1_h = _sub_rn_f32(p1_h.to(tl.float32), p2_h.to(tl.float32)).to(tl.float16)
        out1_h = out1_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        tl.store(out_ptr + q1_offsets, out1_h, mask=mask,
                 eviction_policy="evict_first")

        p3_h = _mul_rn_f32(q2_f, cos_b).to(tl.float16)
        p4_h = _mul_rn_f32(q1_f, sin_b).to(tl.float16)
        p3_h = p3_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        p4_h = p4_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        out2_h = _add_rn_f32(p3_h.to(tl.float32), p4_h.to(tl.float32)).to(tl.float16)
        out2_h = out2_h.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        tl.store(out_ptr + q2_offsets, out2_h, mask=mask,
                 eviction_policy="evict_first")


def run(q, cos, sin):
    out_dtype = q.dtype
    if q.dtype == torch.float32 or cos.dtype == torch.float32 or sin.dtype == torch.float32:
        out_dtype = torch.float32

    output = torch.empty_like(q, dtype=out_dtype)

    batch_size = q.shape[0]
    seq_len = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    half = head_dim // 2

    BLOCK_H = 32
    BLOCK_D = triton.next_power_of_2(half)
    num_warps = 8
    num_stages = 2
    OUT_FP32 = output.dtype == torch.float32

    grid = (seq_len, triton.cdiv(n_heads, BLOCK_H), batch_size)
    _rope_kernel[grid](
        q, cos, sin, output, seq_len,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        OUT_FP32=OUT_FP32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "RN_ASM": 1,
        "OUT_FP32": int(OUT_FP32),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
