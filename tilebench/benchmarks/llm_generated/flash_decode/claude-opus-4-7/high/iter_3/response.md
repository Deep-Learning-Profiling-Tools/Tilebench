Looking at the trajectory:
- iter 1 (best): single-kernel approach with D_SPLITS=4 → 64 CTAs (5.7% / 8.2%)
- iter 2 (regressed): 2-stage K-split approach (3.5% / 3.9%)

The bottleneck is SM utilization — only 64 CTAs on a 132-SM B200. Iter 2's 2-stage approach added a global-memory round-trip that hurt more than the extra parallelism helped.

My approach for iter 3: stay with iter 1's single-kernel D-split structure (no merge kernel overhead), but **double the parallelism** by halving `BLOCK_D` (16 instead of 32) → 128 CTAs, ~1 per SM, while keeping memory coalescing reasonable (16 fp32 = 64B per row). Reduce `num_warps` to 2 to keep occupancy room.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_decode_kernel(
    mid_o_ptr, mid_o_lse_ptr, b_seqlen_ptr, output_ptr,
    stride_mb, stride_mh, stride_mk, stride_md,
    stride_lb, stride_lh, stride_lk,
    stride_ob, stride_oh, stride_od,
    block_seq, NUM_BLOCKS, HEAD_DIM,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d_block = tl.program_id(2)

    seqlen = tl.load(b_seqlen_ptr + b)
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    lse_row_off = b * stride_lb + h * stride_lh
    mid_o_row_off = b * stride_mb + h * stride_mh

    # Pass 1: global max over LSE
    m = -float('inf')
    for kk in range(0, NUM_BLOCKS, BLOCK_K):
        k_offs = kk + offs_k
        mask = k_offs < valid_blocks
        lse = tl.load(mid_o_lse_ptr + lse_row_off + k_offs * stride_lk,
                      mask=mask, other=-float('inf')).to(tl.float32)
        m = tl.maximum(m, tl.max(lse, axis=0))

    # Pass 2: weighted sum
    numer = tl.zeros((BLOCK_D,), dtype=tl.float32)
    denom = 0.0
    for kk in range(0, NUM_BLOCKS, BLOCK_K):
        k_offs = kk + offs_k
        mask_k = k_offs < valid_blocks
        lse = tl.load(mid_o_lse_ptr + lse_row_off + k_offs * stride_lk,
                      mask=mask_k, other=-float('inf')).to(tl.float32)
        w = tl.exp(lse - m)
        w = tl.where(mask_k, w, 0.0)
        denom = denom + tl.sum(w, axis=0)

        ptrs = (mid_o_ptr + mid_o_row_off
                + k_offs[:, None] * stride_mk
                + offs_d[None, :] * stride_md)
        full_mask = mask_k[:, None] & d_mask[None, :]
        vals = tl.load(ptrs, mask=full_mask, other=0.0).to(tl.float32)
        numer = numer + tl.sum(vals * w[:, None], axis=0)

    result = numer / denom
    out_ptrs = output_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, result.to(output_ptr.dtype.element_ty), mask=d_mask)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_D = 16
    BLOCK_K = 128
    num_warps = 2
    num_stages = 3

    D_SPLITS = triton.cdiv(head_dim, BLOCK_D)
    grid = (batch, heads, D_SPLITS)
    _flash_decode_kernel[grid](
        mid_o, mid_o_lse, b_seqlen, output,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_lse.stride(0), mid_o_lse.stride(1), mid_o_lse.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_seq, num_blocks, head_dim,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D, "D_SPLITS": D_SPLITS,
        "num_warps": num_warps, "num_stages": num_stages,
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


@ct.kernel(occupancy=8)
def _flash_decode_kernel(
    mid_o, mid_o_lse, b_seqlen, output,
    block_seq: int,
    NUM_BLOCKS: ConstInt, HEAD_DIM: ConstInt,
    BLOCK_K: ConstInt, BLOCK_D: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)
    d_block = ct.bid(2)

    seqlen = ct.load(b_seqlen, index=(b,), shape=())
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    num_k_tiles = ct.cdiv(NUM_BLOCKS, BLOCK_K)

    # Pass 1: global max over LSE
    m_running = ct.full((1,), -np.inf, dtype=np.float32)
    for kk in range(num_k_tiles):
        offs_k = ct.arange(BLOCK_K, dtype=np.int32) + kk * BLOCK_K
        mask_k = offs_k < valid_blocks
        lse = ct.load(mid_o_lse, index=(b, h, kk),
                      shape=(1, 1, BLOCK_K),
                      padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_K,))
        lse = ct.astype(lse, np.float32)
        lse = ct.where(mask_k, lse,
                       ct.full((BLOCK_K,), -np.inf, dtype=np.float32))
        m_block = ct.max(lse, axis=0, keepdims=True)
        m_running = ct.maximum(m_running, m_block)

    # Pass 2: weighted sum for our head_dim slice
    numer = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    denom = ct.full((1,), 0.0, dtype=np.float32)

    for kk in range(num_k_tiles):
        offs_k = ct.arange(BLOCK_K, dtype=np.int32) + kk * BLOCK_K
        mask_k = offs_k < valid_blocks
        lse = ct.load(mid_o_lse, index=(b, h, kk),
                      shape=(1, 1, BLOCK_K),
                      padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_K,))
        lse = ct.astype(lse, np.float32)
        lse = ct.where(mask_k, lse,
                       ct.full((BLOCK_K,), -np.inf, dtype=np.float32))
        w = ct.exp(lse - m_running)
        w = ct.where(mask_k, w,
                     ct.full((BLOCK_K,), 0.0, dtype=np.float32))
        denom = denom + ct.sum(w, axis=0, keepdims=True)

        vals = ct.load(mid_o, index=(b, h, kk, d_block),
                       shape=(1, 1, BLOCK_K, BLOCK_D),
                       padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_K, BLOCK_D))
        vals = ct.astype(vals, np.float32)
        weighted = vals * w[:, None]
        numer = numer + ct.sum(weighted, axis=0)

    result = numer / denom
    out_tile = ct.astype(result, output.dtype).reshape((1, 1, BLOCK_D))
    ct.store(output, index=(b, h, d_block), tile=out_tile)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_D = 16
    BLOCK_K = 128
    D_SPLITS = (head_dim + BLOCK_D - 1) // BLOCK_D
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (batch, heads, D_SPLITS)
    ct.launch(stream, grid, _flash_decode_kernel,
              (mid_o, mid_o_lse, b_seqlen, output,
               block_seq, num_blocks, head_dim,
               BLOCK_K, BLOCK_D))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D,
                      "D_SPLITS": D_SPLITS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverting to iter 1's single-kernel D-split structure but doubling parallelism: `BLOCK_D=16` (D_SPLITS=8) → 128 CTAs ≈ 1 per SM on B200 (vs 64 in iter 1), with `num_warps=2` and `num_stages=3` for deeper load pipelining. Avoids iter 2's costly 2-stage merge round-trip.
