Looking at the trajectory:
- **Triton best** was iter 5 (7.2%): two-pass split-K with BLOCK_K=64, K_SPLITS=5, BLOCK_D=16, D_SPLITS=8. Iter 6 regressed.
- **CuTile best** was iter 3 (8.3%): single-pass with BLOCK_K=128, BLOCK_D=16, D_SPLITS=8. Iter 6 regressed.

Reverting to best approaches with new optimizations:
- **Triton**: Keep two-pass split-K but increase BLOCK_K=128 → K_SPLITS=3 (fewer partial outputs to write+read in stage 2)
- **CuTile**: Keep single-pass but use BLOCK_K=256 → only 2 K iterations instead of 3 (less loop overhead)

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
    D_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sd = tl.program_id(2)
    s = sd // D_SPLITS
    d_block = sd % D_SPLITS

    seqlen = tl.load(b_seqlen_ptr + b)
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    k_start = s * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < valid_blocks

    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
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
    d_block = tl.program_id(2)

    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
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

    BLOCK_D = 16
    BLOCK_K = 128
    num_warps = 2
    num_stages = 2

    D_SPLITS = triton.cdiv(head_dim, BLOCK_D)
    K_SPLITS = triton.cdiv(num_blocks, BLOCK_K)

    p_m = torch.empty((batch, heads, K_SPLITS), device=mid_o.device, dtype=torch.float32)
    p_d = torch.empty((batch, heads, K_SPLITS), device=mid_o.device, dtype=torch.float32)
    p_o = torch.empty((batch, heads, K_SPLITS, head_dim),
                      device=mid_o.device, dtype=torch.float32)

    grid1 = (batch, heads, K_SPLITS * D_SPLITS)
    _stage1_kernel[grid1](
        mid_o, mid_o_lse, b_seqlen, p_m, p_d, p_o,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_lse.stride(0), mid_o_lse.stride(1), mid_o_lse.stride(2),
        p_m.stride(0), p_m.stride(1), p_m.stride(2),
        p_o.stride(0), p_o.stride(1), p_o.stride(2), p_o.stride(3),
        block_seq, num_blocks, head_dim,
        D_SPLITS=D_SPLITS,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    BLOCK_S = max(triton.next_power_of_2(K_SPLITS), 8)
    grid2 = (batch, heads, D_SPLITS)
    _stage2_kernel[grid2](
        p_m, p_d, p_o, output,
        p_m.stride(0), p_m.stride(1), p_m.stride(2),
        p_o.stride(0), p_o.stride(1), p_o.stride(2), p_o.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        head_dim,
        K_SPLITS=K_SPLITS, BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
        num_warps=1, num_stages=1,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D,
        "D_SPLITS": D_SPLITS, "K_SPLITS": K_SPLITS,
        "BLOCK_S": BLOCK_S,
        "num_warps": num_warps, "num_stages": num_stages,
        "two_pass_ksplit": True,
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
    BLOCK_K = 256  # fewer K iterations than iter 3's 128
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

Reverted both backends to their best verify-clean configurations (Triton iter 5, CuTile iter 3) and applied a single new optimization: increased BLOCK_K to reduce K-loop iterations (Triton: 64→128 yielding K_SPLITS=3, CuTile: 128→256 yielding 2 K-loop iters), reducing partial output traffic for Triton and loop overhead for CuTile.
