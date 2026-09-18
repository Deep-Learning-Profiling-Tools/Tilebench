Looking at the trajectory:
- Best cuTile (iter 3) uses single-pass with D_SPLITS=8 → reads mid_o 8× (bad for bandwidth)
- Best Triton (iter 5) uses two-pass K-split + D_SPLITS=8 → still reads mid_o 8× per K-split

This is a bandwidth-bound problem (~2.5 MB of mid_o). The key insight: **don't split D, only split K**. Each stage-1 CTA owns full head_dim for a K-chunk, so mid_o is read exactly once. With BLOCK_K=16, K_SPLITS=20, batch*heads*K_SPLITS=320 CTAs gives ~2.2× SM occupancy on B200's 148 SMs while keeping the tile small enough to avoid register spills.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _stage1_kernel(
    mid_o_ptr, mid_o_lse_ptr, b_seqlen_ptr,
    p_m_ptr, p_d_ptr, p_o_ptr,
    stride_mb, stride_mh, stride_mk, stride_md,
    stride_lb, stride_lh, stride_lk,
    stride_pmb, stride_pmh, stride_pms,
    stride_pob, stride_poh, stride_pos, stride_pod,
    block_seq, NUM_BLOCKS, HEAD_DIM,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    seqlen = tl.load(b_seqlen_ptr + b)
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    k_start = s * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < valid_blocks
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    lse = tl.load(mid_o_lse_ptr + b * stride_lb + h * stride_lh + offs_k * stride_lk,
                  mask=mask_k, other=-float('inf')).to(tl.float32)

    vals = tl.load(mid_o_ptr + b * stride_mb + h * stride_mh
                   + offs_k[:, None] * stride_mk + offs_d[None, :] * stride_md,
                   mask=mask_k[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    m_local = tl.max(lse, axis=0)
    m_safe = tl.where(m_local == -float('inf'), 0.0, m_local)
    w = tl.exp(lse - m_safe)
    w = tl.where(mask_k, w, 0.0)
    d_local = tl.sum(w, axis=0)
    o_local = tl.sum(vals * w[:, None], axis=0)

    tl.store(p_m_ptr + b * stride_pmb + h * stride_pmh + s * stride_pms, m_local)
    tl.store(p_d_ptr + b * stride_pmb + h * stride_pmh + s * stride_pms, d_local)
    tl.store(p_o_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_d * stride_pod,
             o_local, mask=d_mask)


@triton.jit
def _stage2_kernel(
    p_m_ptr, p_d_ptr, p_o_ptr, output_ptr,
    stride_pmb, stride_pmh, stride_pms,
    stride_pob, stride_poh, stride_pos, stride_pod,
    stride_ob, stride_oh, stride_od,
    HEAD_DIM,
    K_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < K_SPLITS

    m_local = tl.load(p_m_ptr + b * stride_pmb + h * stride_pmh + offs_s * stride_pms,
                      mask=mask_s, other=-float('inf'))
    d_local = tl.load(p_d_ptr + b * stride_pmb + h * stride_pmh + offs_s * stride_pms,
                      mask=mask_s, other=0.0)
    o_local = tl.load(p_o_ptr + b * stride_pob + h * stride_poh
                      + offs_s[:, None] * stride_pos + offs_d[None, :] * stride_pod,
                      mask=mask_s[:, None] & d_mask[None, :], other=0.0)

    m_star = tl.max(m_local, axis=0)
    m_star_safe = tl.where(m_star == -float('inf'), 0.0, m_star)
    alpha = tl.exp(m_local - m_star_safe)
    valid = mask_s & (m_local != -float('inf'))
    alpha = tl.where(valid, alpha, 0.0)

    new_d = tl.sum(d_local * alpha, axis=0)
    new_o = tl.sum(o_local * alpha[:, None], axis=0)

    result = new_o / new_d
    tl.store(output_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od,
             result.to(output_ptr.dtype.element_ty), mask=d_mask)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 16
    BLOCK_D = triton.next_power_of_2(head_dim)
    num_warps = 4
    num_stages = 2

    K_SPLITS = triton.cdiv(num_blocks, BLOCK_K)

    p_m = torch.empty((batch, heads, K_SPLITS), device=mid_o.device, dtype=torch.float32)
    p_d = torch.empty((batch, heads, K_SPLITS), device=mid_o.device, dtype=torch.float32)
    p_o = torch.empty((batch, heads, K_SPLITS, head_dim),
                      device=mid_o.device, dtype=torch.float32)

    grid1 = (batch, heads, K_SPLITS)
    _stage1_kernel[grid1](
        mid_o, mid_o_lse, b_seqlen, p_m, p_d, p_o,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_lse.stride(0), mid_o_lse.stride(1), mid_o_lse.stride(2),
        p_m.stride(0), p_m.stride(1), p_m.stride(2),
        p_o.stride(0), p_o.stride(1), p_o.stride(2), p_o.stride(3),
        block_seq, num_blocks, head_dim,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    BLOCK_S = max(triton.next_power_of_2(K_SPLITS), 8)
    grid2 = (batch, heads)
    _stage2_kernel[grid2](
        p_m, p_d, p_o, output,
        p_m.stride(0), p_m.stride(1), p_m.stride(2),
        p_o.stride(0), p_o.stride(1), p_o.stride(2), p_o.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        head_dim,
        K_SPLITS=K_SPLITS, BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
        num_warps=2, num_stages=1,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D,
        "K_SPLITS": K_SPLITS, "BLOCK_S": BLOCK_S,
        "num_warps": num_warps, "num_stages": num_stages,
        "D_SPLITS": 1,
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
def _stage1_kernel(
    mid_o, mid_o_lse, b_seqlen,
    p_m, p_d, p_o,
    block_seq: int,
    NUM_BLOCKS: ConstInt, HEAD_DIM: ConstInt,
    BLOCK_K: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)

    seqlen = ct.load(b_seqlen, index=(b,), shape=())
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    offs_k = ct.arange(BLOCK_K, dtype=np.int32) + s * BLOCK_K
    mask_k = offs_k < valid_blocks

    lse = ct.load(mid_o_lse, index=(b, h, s),
                  shape=(1, 1, BLOCK_K),
                  padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_K,))
    lse = ct.astype(lse, np.float32)
    lse = ct.where(mask_k, lse,
                   ct.full((BLOCK_K,), -np.inf, dtype=np.float32))

    vals = ct.load(mid_o, index=(b, h, s, 0),
                   shape=(1, 1, BLOCK_K, HEAD_DIM),
                   padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_K, HEAD_DIM))
    vals = ct.astype(vals, np.float32)

    m_local = ct.max(lse, axis=0, keepdims=True)
    m_safe = ct.where(m_local == -np.inf,
                      ct.full((1,), 0.0, dtype=np.float32),
                      m_local)
    w = ct.exp(lse - m_safe)
    w = ct.where(mask_k, w, ct.full((BLOCK_K,), 0.0, dtype=np.float32))

    d_local = ct.sum(w, axis=0, keepdims=True)
    o_local = ct.sum(vals * w[:, None], axis=0)

    ct.store(p_m, index=(b, h, s), tile=m_local.reshape((1, 1, 1)))
    ct.store(p_d, index=(b, h, s), tile=d_local.reshape((1, 1, 1)))
    ct.store(p_o, index=(b, h, s, 0),
             tile=o_local.reshape((1, 1, 1, HEAD_DIM)))


@ct.kernel(occupancy=8)
def _stage2_kernel(
    p_m, p_d, p_o, output,
    HEAD_DIM: ConstInt,
    K_SPLITS: ConstInt,
    BLOCK_S: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)

    offs_s = ct.arange(BLOCK_S, dtype=np.int32)
    mask_s = offs_s < K_SPLITS

    m_local = ct.load(p_m, index=(b, h, 0),
                      shape=(1, 1, BLOCK_S),
                      padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_S,))
    d_local = ct.load(p_d, index=(b, h, 0),
                      shape=(1, 1, BLOCK_S),
                      padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_S,))
    o_local = ct.load(p_o, index=(b, h, 0, 0),
                      shape=(1, 1, BLOCK_S, HEAD_DIM),
                      padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_S, HEAD_DIM))

    m_local = ct.where(mask_s, m_local,
                       ct.full((BLOCK_S,), -np.inf, dtype=np.float32))

    m_star = ct.max(m_local, axis=0, keepdims=True)
    m_star_safe = ct.where(m_star == -np.inf,
                           ct.full((1,), 0.0, dtype=np.float32),
                           m_star)

    alpha = ct.exp(m_local - m_star_safe)
    valid = mask_s & (m_local != -np.inf)
    alpha = ct.where(valid, alpha,
                     ct.full((BLOCK_S,), 0.0, dtype=np.float32))

    new_d = ct.sum(d_local * alpha, axis=0, keepdims=True)
    new_o = ct.sum(o_local * alpha[:, None], axis=0)

    result = new_o / new_d
    out_tile = ct.astype(result, output.dtype).reshape((1, 1, HEAD_DIM))
    ct.store(output, index=(b, h, 0), tile=out_tile)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 16
    HEAD_DIM_P2 = _next_pow2(head_dim)

    K_SPLITS = (num_blocks + BLOCK_K - 1) // BLOCK_K

    p_m = torch.empty((batch, heads, K_SPLITS),
                      device=mid_o.device, dtype=torch.float32)
    p_d = torch.empty((batch, heads, K_SPLITS),
                      device=mid_o.device, dtype=torch.float32)
    p_o = torch.empty((batch, heads, K_SPLITS, head_dim),
                      device=mid_o.device, dtype=torch.float32)

    BLOCK_S = max(_next_pow2(K_SPLITS), 8)

    stream = torch.cuda.current_stream()

    grid1 = (batch, heads, K_SPLITS)
    ct.launch(stream, grid1, _stage1_kernel,
              (mid_o, mid_o_lse, b_seqlen, p_m, p_d, p_o,
               block_seq, num_blocks, HEAD_DIM_P2, BLOCK_K))

    grid2 = (batch, heads, 1)
    ct.launch(stream, grid2, _stage2_kernel,
              (p_m, p_d, p_o, output,
               HEAD_DIM_P2, K_SPLITS, BLOCK_S))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": HEAD_DIM_P2,
        "K_SPLITS": K_SPLITS, "BLOCK_S": BLOCK_S,
        "D_SPLITS": 1, "occupancy": 4,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Approach**: drop D_SPLITS entirely so each stage-1 CTA reads `mid_o[b,h,k_chunk,:]` exactly once (full head_dim in a single tile). With BLOCK_K=16, K_SPLITS=20, that gives 320 stage-1 CTAs on 148 SMs (~2.2× occupancy) reading mid_o exactly once — bandwidth-optimal. Stage 2 then collapses 20 K-partials over 16 (b,h) CTAs.
