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

    # Pass 1: local max for the K group.
    m_i = ct.full((), -np.inf, dtype=np.float32)
    for kk in range(0, BLOCK_K):
        k = g * BLOCK_K + kk
        valid = (k < NUM_BLOCKS) & (k < valid_blocks)

        cur_lse_raw = ct.astype(
            ct.load(
                mid_o_lse,
                index=(b, h, k),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )
        cur_lse = ct.where(valid, cur_lse_raw, -np.inf)
        m_i = ct.maximum(m_i, cur_lse)

    # Pass 2: numerator and denominator on the fixed local max scale.
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)
    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for kk in range(0, BLOCK_K):
        k = g * BLOCK_K + kk
        valid = (k < NUM_BLOCKS) & (k < valid_blocks)

        cur_lse_raw = ct.astype(
            ct.load(
                mid_o_lse,
                index=(b, h, k),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )
        cur_lse = ct.where(valid, cur_lse_raw, -np.inf)
        beta = ct.where(valid, ct.exp(cur_lse - m_safe), 0.0)

        vals = ct.astype(
            ct.load(
                mid_o,
                index=(b, h, k, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc + vals * beta
        l_i = l_i + beta

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

    # Pass 1: global max across group maxima.  BLOCK_G is a power-of-two
    # envelope; OOB group loads are padded to neutral values.
    m_i = ct.full((), -np.inf, dtype=np.float32)
    for gg in range(0, BLOCK_G):
        gm = ct.astype(
            ct.load(
                tmp_m,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )
        m_i = ct.maximum(m_i, gm)

    # Pass 2: rescale each group partial to the global max scale.
    m_safe = ct.where(m_i == -np.inf, 0.0, m_i)
    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for gg in range(0, BLOCK_G):
        gm = ct.astype(
            ct.load(
                tmp_m,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
            ),
            np.float32,
        )
        gl = ct.astype(
            ct.load(
                tmp_l,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ),
            np.float32,
        )

        beta = ct.where(gl > 0.0, ct.exp(gm - m_safe), 0.0)

        group_acc = ct.astype(
            ct.load(
                tmp_acc,
                index=(b, h, gg, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc + group_acc * beta
        l_i = l_i + gl * beta

    out = acc / (l_i + 1.0e-10)
    ct.store(output, index=(b, h, dchunk), tile=out.reshape((1, 1, BLOCK_D)))


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 128
    BLOCK_G = 16

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

    grid_partial = (num_groups, batch * heads, num_dchunks)
    ct.launch(
        stream,
        grid_partial,
        _partial_flash_decode_kernel,
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
        _final_flash_decode_kernel,
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
            "two_pass_merge": 1,
            "block_seq": block_seq_value,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
