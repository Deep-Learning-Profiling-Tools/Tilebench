```python title="impl_triton.py"
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
                 eviction_policy="evict_first").to(tl.float32)
    q2 = tl.load(q_ptr + q2_offsets, mask=mask, other=0.0,
                 eviction_policy="evict_first").to(tl.float32)

    cos_v = tl.load(cos_ptr + trig_offsets, mask=mask_d, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
    sin_v = tl.load(sin_ptr + trig_offsets, mask=mask_d, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)

    # Match eager PyTorch fp16 semantics: each multiply materializes an
    # intermediate tensor in output dtype before the add/sub kernel.
    p1 = (q1 * cos_v[None, :]).to(q_ptr.dtype.element_ty).to(tl.float32)
    p2 = (q2 * sin_v[None, :]).to(q_ptr.dtype.element_ty).to(tl.float32)
    out1 = p1 - p2
    tl.store(out_ptr + q1_offsets, out1, mask=mask, eviction_policy="evict_first")

    p3 = (q2 * cos_v[None, :]).to(q_ptr.dtype.element_ty).to(tl.float32)
    p4 = (q1 * sin_v[None, :]).to(q_ptr.dtype.element_ty).to(tl.float32)
    out2 = p3 + p4
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel(q, cos, sin, output,
                 HEAD_DIM: ConstInt, BLOCK_H: ConstInt, BLOCK_D: ConstInt):
    s = ct.bid(0)
    hb = ct.bid(1)
    b = ct.bid(2)

    half = HEAD_DIM // 2

    q_first = q.slice(3, 0, half)
    q_second = q.slice(3, half, HEAD_DIM)
    out_first = output.slice(3, 0, half)
    out_second = output.slice(3, half, HEAD_DIM)

    q1 = ct.load(
        q_first,
        index=(b, s, hb, 0),
        shape=(1, 1, BLOCK_H, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    q2 = ct.load(
        q_second,
        index=(b, s, hb, 0),
        shape=(1, 1, BLOCK_H, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    cos_tile = ct.load(
        cos,
        index=(s, 0),
        shape=(1, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    sin_tile = ct.load(
        sin,
        index=(s, 0),
        shape=(1, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    cos4 = ct.reshape(cos_tile, (1, 1, 1, BLOCK_D))
    sin4 = ct.reshape(sin_tile, (1, 1, 1, BLOCK_D))

    q1f = ct.astype(q1, np.float32)
    q2f = ct.astype(q2, np.float32)
    cosf = ct.astype(cos4, np.float32)
    sinf = ct.astype(sin4, np.float32)

    p1 = ct.astype(q1f * cosf, output.dtype)
    p2 = ct.astype(q2f * sinf, output.dtype)
    out1 = ct.astype(p1, np.float32) - ct.astype(p2, np.float32)
    ct.store(
        out_first,
        index=(b, s, hb, 0),
        tile=ct.astype(out1, output.dtype),
        latency=1,
        allow_tma=False,
    )

    p3 = ct.astype(q2f * cosf, output.dtype)
    p4 = ct.astype(q1f * sinf, output.dtype)
    out2 = ct.astype(p3, np.float32) + ct.astype(p4, np.float32)
    ct.store(
        out_second,
        index=(b, s, hb, 0),
        tile=ct.astype(out2, output.dtype),
        latency=1,
        allow_tma=False,
    )


def run(q, cos, sin):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    seq_len = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    half = head_dim // 2

    BLOCK_H = 32
    BLOCK_D = 1 << (half - 1).bit_length()
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (seq_len, ct.cdiv(n_heads, BLOCK_H), batch_size)
    ct.launch(stream, grid, _rope_kernel, (q, cos, sin, output, head_dim, BLOCK_H, BLOCK_D))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_D": BLOCK_D,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
