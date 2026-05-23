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
    # Grid order: (D_SPLITS, heads, batch) so adjacent CTAs share (b,h)
    d_block = ct.bid(0)
    h = ct.bid(1)
    b = ct.bid(2)

    seqlen = ct.load(b_seqlen, index=(b,), shape=())
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    num_k_tiles = ct.cdiv(NUM_BLOCKS, BLOCK_K)

    m_i = ct.full((1,), -np.inf, dtype=np.float32)
    l_i = ct.full((1,), 0.0, dtype=np.float32)
    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)

    for kk in range(num_k_tiles):
        offs_k = ct.arange(BLOCK_K, dtype=np.int32) + kk * BLOCK_K
        mask_k = offs_k < valid_blocks

        lse = ct.load(mid_o_lse, index=(b, h, kk),
                      shape=(1, 1, BLOCK_K),
                      padding_mode=ct.PaddingMode.NEG_INF).reshape((BLOCK_K,))
        lse = ct.astype(lse, np.float32)
        lse = ct.where(mask_k, lse,
                       ct.full((BLOCK_K,), -np.inf, dtype=np.float32))

        m_block = ct.max(lse, keepdims=True)        # shape (1,)
        m_new = ct.maximum(m_i, m_block)            # shape (1,)
        alpha = ct.exp(m_i - m_new)                 # shape (1,)
        w = ct.exp(lse - m_new)                     # shape (BLOCK_K,)
        w = ct.where(mask_k, w,
                     ct.full((BLOCK_K,), 0.0, dtype=np.float32))

        vals = ct.load(mid_o, index=(b, h, kk, d_block),
                       shape=(1, 1, BLOCK_K, BLOCK_D),
                       padding_mode=ct.PaddingMode.ZERO).reshape((BLOCK_K, BLOCK_D))
        vals = ct.astype(vals, np.float32)

        acc = acc * alpha + ct.sum(vals * w[:, None], axis=0)
        l_i = l_i * alpha + ct.sum(w, keepdims=True)
        m_i = m_new

    result = acc / l_i
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
    BLOCK_K = 256
    D_SPLITS = (head_dim + BLOCK_D - 1) // BLOCK_D
    occupancy = 8

    stream = torch.cuda.current_stream()
    # Put D_SPLITS as axis 0 so adjacent CTAs share (b,h) -> better L2 reuse
    grid = (D_SPLITS, heads, batch)
    ct.launch(stream, grid, _flash_decode_kernel,
              (mid_o, mid_o_lse, b_seqlen, output,
               block_seq, num_blocks, head_dim,
               BLOCK_K, BLOCK_D))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D,
                      "D_SPLITS": D_SPLITS, "occupancy": occupancy,
                      "single_pass_online": True,
                      "grid_order": "d_first"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
