import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


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
        # Compute in fp32 but round each multiply to fp16 between ops to
        # bit-match torch eager (which produces fp16 intermediates).
        q1f = q1.to(tl.float32)
        q2f = q2.to(tl.float32)
        cf = cb.to(tl.float32)
        snf = snb.to(tl.float32)
        t1 = (q1f * cf).to(tl.float16).to(tl.float32)
        t2 = (q2f * snf).to(tl.float16).to(tl.float32)
        out1 = (t1 - t2).to(tl.float16)
        t3 = (q2f * cf).to(tl.float16).to(tl.float32)
        t4 = (q1f * snf).to(tl.float16).to(tl.float32)
        out2 = (t3 + t4).to(tl.float16)
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

    is_fp16 = (q.dtype == torch.float16) and (out_dtype == torch.float16)

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
