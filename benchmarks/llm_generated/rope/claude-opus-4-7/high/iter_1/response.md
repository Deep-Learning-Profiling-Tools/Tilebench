Looking at the iter 0 results:
- **Triton**: ~100% roofline but verify failed with abs diff 0.0078 (fp16). Issue: doing math in fp32 then downcasting gives slightly different rounding than torch reference which does math in source dtype.
- **cuTile**: `.with_hints()` doesn't exist — occupancy must be set via `@ct.kernel(occupancy=...)` decorator.

Fix: compute in source dtype to match torch reference bit-equivalently (within tolerance); for cuTile, use decorator-level occupancy hint.

```python title="impl_triton.py"
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
                 BLOCK_D: tl.constexpr):
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

    # Compute in source dtype to match torch reference rounding bit-equivalently
    out1 = q1 * c[None, :] - q2 * sn[None, :]
    out2 = q2 * c[None, :] + q1 * sn[None, :]

    o_base = out_ptr + b * sob + s * sos
    o1_ptrs = o_base + offs_h[:, None] * soh + offs_d[None, :] * sod
    o2_ptrs = o1_ptrs + HALF * sod
    tl.store(o1_ptrs, out1, mask=mask)
    tl.store(o2_ptrs, out2, mask=mask)


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    output = torch.empty_like(q)

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(half)

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
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_H": BLOCK_H, "BLOCK_D": BLOCK_D,
                      "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel(q, cos, sin, output,
                 S: ConstInt, H: ConstInt, HALF: ConstInt):
    pid = ct.bid(0)
    b = pid // S
    s = pid - b * S

    q1 = ct.load(q, index=(b, s, 0, 0), shape=(1, 1, H, HALF)).reshape((H, HALF))
    q2 = ct.load(q, index=(b, s, 0, 1), shape=(1, 1, H, HALF)).reshape((H, HALF))

    c = ct.load(cos, index=(s, 0), shape=(1, HALF))
    sn = ct.load(sin, index=(s, 0), shape=(1, HALF))

    out1 = q1 * c - q2 * sn
    out2 = q2 * c + q1 * sn

    ct.store(output, index=(b, s, 0, 0), tile=out1.reshape((1, 1, H, HALF)))
    ct.store(output, index=(b, s, 0, 1), tile=out2.reshape((1, 1, H, HALF)))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    output = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    occupancy = 4

    grid = (B * S, 1, 1)
    ct.launch(stream, grid, _rope_kernel, (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
