import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _stage1_kernel(
    mid_o, mid_o_lse, b_seqlen,
    p_m, p_d, p_o,
    block_seq: int,
    NUM_BLOCKS: ConstInt, HEAD_DIM: ConstInt,
    D_SPLITS: ConstInt,
    BLOCK_K: ConstInt, BLOCK_D: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)
    sd = ct.bid(2)
    s = sd // D_SPLITS
    d_block = sd % D_SPLITS

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

    vals = ct.load(mid_o, index=(b, h, s, d_block),
                   shape=(1, 1, BLOCK_K, BLOCK_D),
                   padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_K, BLOCK_D))
    vals = ct.astype(vals, np.float32)

    m_local = ct.max(lse, axis=0, keepdims=True)  # (1,)
    m_safe = ct.where(m_local == -np.inf,
                      ct.full((1,), 0.0, dtype=np.float32),
                      m_local)
    w = ct.exp(lse - m_safe)
    w = ct.where(mask_k, w, ct.full((BLOCK_K,), 0.0, dtype=np.float32))

    d_local = ct.sum(w, axis=0, keepdims=True)  # (1,)
    o_local = ct.sum(vals * w[:, None], axis=0)  # (BLOCK_D,)

    # All D_SPLITS CTAs write the same m_local/d_local (idempotent).
    ct.store(p_m, index=(b, h, s), tile=m_local.reshape((1, 1, 1)))
    ct.store(p_d, index=(b, h, s), tile=d_local.reshape((1, 1, 1)))
    ct.store(p_o, index=(b, h, s, d_block),
             tile=o_local.reshape((1, 1, 1, BLOCK_D)))


@ct.kernel(occupancy=8)
def _stage2_kernel(
    p_m, p_d, p_o, output,
    HEAD_DIM: ConstInt,
    K_SPLITS: ConstInt,
    BLOCK_D: ConstInt, BLOCK_S: ConstInt,
):
    b = ct.bid(0)
    h = ct.bid(1)
    d_block = ct.bid(2)

    offs_s = ct.arange(BLOCK_S, dtype=np.int32)
    mask_s = offs_s < K_SPLITS

    m_local = ct.load(p_m, index=(b, h, 0),
                      shape=(1, 1, BLOCK_S),
                      padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_S,))
    d_local = ct.load(p_d, index=(b, h, 0),
                      shape=(1, 1, BLOCK_S),
                      padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_S,))
    o_local = ct.load(p_o, index=(b, h, 0, d_block),
                      shape=(1, 1, BLOCK_S, BLOCK_D),
                      padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_S, BLOCK_D))

    m_local = ct.where(mask_s, m_local,
                       ct.full((BLOCK_S,), -np.inf, dtype=np.float32))

    m_star = ct.max(m_local, axis=0, keepdims=True)  # (1,)
    m_star_safe = ct.where(m_star == -np.inf,
                           ct.full((1,), 0.0, dtype=np.float32),
                           m_star)

    alpha = ct.exp(m_local - m_star_safe)
    valid = mask_s & (m_local != -np.inf)
    alpha = ct.where(valid, alpha,
                     ct.full((BLOCK_S,), 0.0, dtype=np.float32))

    new_d = ct.sum(d_local * alpha, axis=0, keepdims=True)  # (1,)
    new_o = ct.sum(o_local * alpha[:, None], axis=0)         # (BLOCK_D,)

    result = new_o / new_d  # (BLOCK_D,)
    out_tile = ct.astype(result, output.dtype).reshape((1, 1, BLOCK_D))
    ct.store(output, index=(b, h, d_block), tile=out_tile)


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

    BLOCK_D = 16
    BLOCK_K = 64

    D_SPLITS = (head_dim + BLOCK_D - 1) // BLOCK_D
    K_SPLITS = (num_blocks + BLOCK_K - 1) // BLOCK_K

    p_m = torch.empty((batch, heads, K_SPLITS),
                      device=mid_o.device, dtype=torch.float32)
    p_d = torch.empty((batch, heads, K_SPLITS),
                      device=mid_o.device, dtype=torch.float32)
    p_o = torch.empty((batch, heads, K_SPLITS, head_dim),
                      device=mid_o.device, dtype=torch.float32)

    BLOCK_S = max(_next_pow2(K_SPLITS), 8)

    stream = torch.cuda.current_stream()

    grid1 = (batch, heads, K_SPLITS * D_SPLITS)
    ct.launch(stream, grid1, _stage1_kernel,
              (mid_o, mid_o_lse, b_seqlen, p_m, p_d, p_o,
               block_seq, num_blocks, head_dim,
               D_SPLITS, BLOCK_K, BLOCK_D))

    grid2 = (batch, heads, D_SPLITS)
    ct.launch(stream, grid2, _stage2_kernel,
              (p_m, p_d, p_o, output,
               head_dim, K_SPLITS, BLOCK_D, BLOCK_S))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D,
        "D_SPLITS": D_SPLITS, "K_SPLITS": K_SPLITS,
        "BLOCK_S": BLOCK_S,
        "two_pass_ksplit": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
