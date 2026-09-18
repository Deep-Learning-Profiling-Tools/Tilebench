import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _partial_flash_decode_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    tmp_acc,
    tmp_m,
    tmp_l,
    BLOCK_SEQ_VALUE: ConstInt,
    NUM_BLOCKS: ConstInt,
    HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_D: ConstInt,
):
    g = ct.bid(0)
    bh = ct.bid(1)
    dchunk = ct.bid(2)

    b = bh // HEADS
    h = bh - b * HEADS

    seq_len = ct.load(b_seqlen, index=(b,), shape=(), padding_mode=ct.PaddingMode.ZERO)
    valid_blocks = (seq_len + BLOCK_SEQ_VALUE - 1) // BLOCK_SEQ_VALUE

    offs_k = g * BLOCK_K + ct.arange(BLOCK_K, dtype=np.int32)
    valid = (offs_k < NUM_BLOCKS) & (offs_k < valid_blocks)

    lse_raw = ct.astype(
        ct.load(
            mid_o_lse,
            index=(b, h, g),
            shape=(1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
        ).reshape((BLOCK_K,)),
        np.float32,
    )

    lse = ct.where(valid, lse_raw, -np.inf)
    m_i = ct.max(lse)
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)

    beta = ct.exp2((lse - m_safe) * 1.4426950408889634, flush_to_zero=True)
    beta = ct.where(valid, beta, 0.0)

    vals = ct.astype(
        ct.load(
            mid_o,
            index=(b, h, g, dchunk),
            shape=(1, 1, BLOCK_K, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_K, BLOCK_D)),
        np.float32,
    )

    acc = ct.sum(vals * beta[:, None], axis=0)
    l_i = ct.sum(beta)

    ct.store(tmp_acc, index=(b, h, g, dchunk), tile=acc.reshape((1, 1, 1, BLOCK_D)))
    ct.store(tmp_m, index=(b, h, g), tile=m_i)
    ct.store(tmp_l, index=(b, h, g), tile=l_i)


@ct.kernel
def _final_flash_decode_kernel(
    tmp_acc,
    tmp_m,
    tmp_l,
    output,
    NUM_GROUPS: ConstInt,
    HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_G: ConstInt,
    BLOCK_D: ConstInt,
):
    bh = ct.bid(0)
    dchunk = ct.bid(1)

    b = bh // HEADS
    h = bh - b * HEADS

    gm = ct.astype(
        ct.load(
            tmp_m,
            index=(b, h, 0),
            shape=(1, 1, BLOCK_G),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
        ).reshape((BLOCK_G,)),
        np.float32,
    )

    gl = ct.astype(
        ct.load(
            tmp_l,
            index=(b, h, 0),
            shape=(1, 1, BLOCK_G),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_G,)),
        np.float32,
    )

    group_ids = ct.arange(BLOCK_G, dtype=np.int32)
    valid_g = group_ids < NUM_GROUPS

    gm = ct.where(valid_g, gm, -np.inf)
    gl = ct.where(valid_g, gl, 0.0)

    m_i = ct.max(gm)
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)

    beta = ct.exp2((gm - m_safe) * 1.4426950408889634, flush_to_zero=True)
    beta = ct.where(gl > 0.0, beta, 0.0)

    group_acc = ct.astype(
        ct.load(
            tmp_acc,
            index=(b, h, 0, dchunk),
            shape=(1, 1, BLOCK_G, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
        ).reshape((BLOCK_G, BLOCK_D)),
        np.float32,
    )

    acc = ct.sum(group_acc * beta[:, None], axis=0)
    l_i = ct.sum(gl * beta)

    out = acc / (l_i + 1.0e-10)
    ct.store(output, index=(b, h, dchunk), tile=out.reshape((1, 1, BLOCK_D)))


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 16
    BLOCK_D = 128
    BLOCK_G = 32
    occupancy_partial = 4
    occupancy_final = 2

    if isinstance(block_seq, torch.Tensor):
        block_seq_value = int(block_seq.item())
    else:
        block_seq_value = int(block_seq)

    num_groups = ct.cdiv(num_blocks, BLOCK_K)
    num_dchunks = ct.cdiv(head_dim, BLOCK_D)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    partial_kernel = _partial_flash_decode_kernel.with_hints(occupancy=occupancy_partial)
    final_kernel = _final_flash_decode_kernel.with_hints(occupancy=occupancy_final)

    grid_partial = (num_groups, batch * heads, num_dchunks)
    ct.launch(
        stream,
        grid_partial,
        partial_kernel,
        (
            mid_o,
            mid_o_lse,
            b_seqlen,
            tmp_acc,
            tmp_m,
            tmp_l,
            block_seq_value,
            num_blocks,
            heads,
            head_dim,
            BLOCK_K,
            BLOCK_D,
        ),
    )

    grid_final = (batch * heads, num_dchunks, 1)
    ct.launch(
        stream,
        grid_final,
        final_kernel,
        (
            tmp_acc,
            tmp_m,
            tmp_l,
            output,
            num_groups,
            heads,
            head_dim,
            BLOCK_G,
            BLOCK_D,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "BLOCK_D": BLOCK_D,
            "BLOCK_G": BLOCK_G,
            "num_groups": num_groups,
            "num_dchunks": num_dchunks,
            "occupancy_partial": occupancy_partial,
            "occupancy_final": occupancy_final,
            "vectorized_reduce": 1,
            "exp2_merge": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
