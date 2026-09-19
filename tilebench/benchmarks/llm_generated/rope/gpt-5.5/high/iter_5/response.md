```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


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

    cos_f = cos_v.to(tl.float32)
    sin_f = sin_v.to(tl.float32)
    q1_f = q1.to(tl.float32)
    q2_f = q2.to(tl.float32)

    if OUT_FP32:
        p1 = q1_f * cos_f[None, :]
        p2 = q2_f * sin_f[None, :]
        # Bitcast round-trip is a no-op numerically, but prevents contraction
        # across PyTorch-eager's materialized multiply temporaries.
        p1 = p1.to(tl.uint32, bitcast=True).to(tl.float32, bitcast=True)
        p2 = p2.to(tl.uint32, bitcast=True).to(tl.float32, bitcast=True)
        out1 = p1 - p2
        tl.store(out_ptr + q1_offsets, out1, mask=mask,
                 eviction_policy="evict_first")

        p1 = q2_f * cos_f[None, :]
        p2 = q1_f * sin_f[None, :]
        p1 = p1.to(tl.uint32, bitcast=True).to(tl.float32, bitcast=True)
        p2 = p2.to(tl.uint32, bitcast=True).to(tl.float32, bitcast=True)
        out2 = p1 + p2
        tl.store(out_ptr + q2_offsets, out2, mask=mask,
                 eviction_policy="evict_first")
    else:
        # Match CUDA eager fp16 TensorIterator semantics: each multiply writes a
        # half-precision temporary, then add/sub produces the half output.
        p1 = (q1_f * cos_f[None, :]).to(tl.float16)
        p2 = (q2_f * sin_f[None, :]).to(tl.float16)
        p1 = p1.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        p2 = p2.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        out1 = (p1.to(tl.float32) - p2.to(tl.float32)).to(tl.float16)
        out1 = out1.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        tl.store(out_ptr + q1_offsets, out1, mask=mask,
                 eviction_policy="evict_first")

        p1 = (q2_f * cos_f[None, :]).to(tl.float16)
        p2 = (q1_f * sin_f[None, :]).to(tl.float16)
        p1 = p1.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        p2 = p2.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        out2 = (p1.to(tl.float32) + p2.to(tl.float32)).to(tl.float16)
        out2 = out2.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True)
        tl.store(out_ptr + q2_offsets, out2, mask=mask,
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
        "PRODUCT_BARRIER": 1,
        "OUT_FP32": int(OUT_FP32),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
