Looking at this problem, I'll parallelize over `(batch * seq_len)` with each program handling all `n_heads` and the full `half = head_dim/2` slice in one tile. This gives ~2048-4096 elements per program — good for memory throughput with cos/sin reused across all heads.

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
    c = tl.load(cos_ptr + cs_off, mask=mask_d, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + cs_off, mask=mask_d, other=0.0).to(tl.float32)

    q1 = tl.load(q1_ptrs, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q2_ptrs, mask=mask, other=0.0).to(tl.float32)

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


@ct.kernel
def _rope_kernel(q, cos, sin, output,
                 S: ConstInt, H: ConstInt, HALF: ConstInt):
    pid = ct.bid(0)
    b = pid // S
    s = pid - b * S

    q1 = ct.astype(ct.load(q, index=(b, s, 0, 0), shape=(1, 1, H, HALF)),
                   np.float32).reshape((H, HALF))
    q2 = ct.astype(ct.load(q, index=(b, s, 0, 1), shape=(1, 1, H, HALF)),
                   np.float32).reshape((H, HALF))

    c = ct.astype(ct.load(cos, index=(s, 0), shape=(1, HALF)), np.float32)
    sn = ct.astype(ct.load(sin, index=(s, 0), shape=(1, HALF)), np.float32)

    out1 = q1 * c - q2 * sn
    out2 = q2 * c + q1 * sn

    ct.store(output, index=(b, s, 0, 0),
             tile=ct.astype(out1.reshape((1, 1, H, HALF)), q.dtype))
    ct.store(output, index=(b, s, 0, 1),
             tile=ct.astype(out2.reshape((1, 1, H, HALF)), q.dtype))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    output = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    occupancy = 4

    grid = (B * S, 1, 1)
    kernel = _rope_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

The approach: one program per `(batch, seq_len)` pair loads the full `(n_heads, half)` slice of q1 and q2 plus a shared `(half,)` cos/sin row, computes the rotation in fp32, and writes back two halves. This gives good arithmetic intensity (cos/sin reused across heads) while keeping the kernel memory-bound on contiguous tiles.
