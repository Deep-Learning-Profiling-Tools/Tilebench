import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _barrier_f16(x):
    # Optimization barrier: PTX `mov.b16` is opaque to the LLVM backend so
    # it cannot contract the producing `mul` and the consuming `sub` into a
    # single FMA. This preserves torch eager's per-op fp16 rounding.
    return tl.inline_asm_elementwise(
        "mov.b16 $0, $1;",
        "=h,h",
        [x],
        dtype=tl.float16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_kernel(q_ptr, cos_ptr, sin_ptr, out_ptr,
                 S, H, HALF,
                 sqb, sqs, sqh, sqd,
                 scs, scd,
                 sob, sos, soh, sod,
                 BLOCK_H: tl.constexpr,
                 BLOCK_D: tl.constexpr,
                 IS_FP16: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < H
    mask_d = offs_d < HALF
    mask = mask_h[:, None] & mask_d[None, :]

    q_base = q_ptr + b * sqb + s * sqs
    q1_ptrs = q_base + offs_h[:, None] * sqh + offs_d[None, :] * sqd
    q2_ptrs = q1_ptrs + HALF * sqd

    cs_off = s * scs + offs_d * scd
    c = tl.load(cos_ptr + cs_off, mask=mask_d, other=0.0)
    sn = tl.load(sin_ptr + cs_off, mask=mask_d, other=0.0)

    q1 = tl.load(q1_ptrs, mask=mask, other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask, other=0.0)

    cb = c[None, :]
    snb = sn[None, :]

    if IS_FP16:
        # All operands are fp16; products are rounded to fp16, then the
        # asm barrier forces the compiler to keep each product as a
        # separate fp16 value before the add/sub (no FMA contraction).
        t1 = _barrier_f16(q1 * cb)
        t2 = _barrier_f16(q2 * snb)
        t3 = _barrier_f16(q2 * cb)
        t4 = _barrier_f16(q1 * snb)
        out1 = t1 - t2
        out2 = t3 + t4
    else:
        out1 = q1 * cb - q2 * snb
        out2 = q2 * cb + q1 * snb

    o_base = out_ptr + b * sob + s * sos
    o1_ptrs = o_base + offs_h[:, None] * soh + offs_d[None, :] * sod
    o2_ptrs = o1_ptrs + HALF * sod
    tl.store(o1_ptrs, out1, mask=mask)
    tl.store(o2_ptrs, out2, mask=mask)


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    out_dtype = torch.result_type(q, cos)
    output = torch.empty(q.shape, dtype=out_dtype, device=q.device)

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(half)

    is_fp16 = (out_dtype == torch.float16)

    num_warps = 4
    num_stages = 2

    grid = (B * S,)
    _rope_kernel[grid](
        q, cos, sin, output,
        S, H, half,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos.stride(0), cos.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        IS_FP16=is_fp16,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_H": BLOCK_H, "BLOCK_D": BLOCK_D,
                      "IS_FP16": int(is_fp16),
                      "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
