import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _partial_flash_decode_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    block_seq_tensor,
    tmp_acc,
    tmp_m,
    tmp_l,
    BLOCK_SEQ_VALUE: ConstInt,
    BLOCK_SEQ_IS_TENSOR: ConstBool,
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

    if BLOCK_SEQ_IS_TENSOR:
        block_seq = ct.load(block_seq_tensor, index=(), shape=(), padding_mode=ct.PaddingMode.ZERO)
    else:
        block_seq = BLOCK_SEQ_VALUE

    seq_len = ct.load(b_seqlen, index=(b,), shape=(), padding_mode=ct.PaddingMode.ZERO)
    valid_blocks = (seq_len + block_seq - 1) // block_seq

    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    m_i = ct.full((), -np.inf, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for kk in range(0, BLOCK_K):
        k = g * BLOCK_K + kk
        valid = (k < NUM_BLOCKS) & (k < valid_blocks)

        cur_lse = ct.astype(
            ct.load(
                mid_o_lse,
                index=(b, h, k),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
            ),
            np.float32,
        )

        cur_m = ct.where(valid, cur_lse, -np.inf)
        m_new = ct.maximum(m_i, cur_m)
        m_safe = ct.where(m_new == -np.inf, 0.0, m_new)

        alpha = ct.where(l_i > 0.0, ct.exp(m_i - m_safe), 0.0)
        beta = ct.where(valid, ct.exp(cur_m - m_safe), 0.0)

        vals = ct.astype(
            ct.load(
                mid_o,
                index=(b, h, k, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc * alpha + vals * beta
        l_i = l_i * alpha + beta
        m_i = m_new

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

    acc = ct.full((BLOCK_D,), 0.0, dtype=np.float32)
    m_i = ct.full((), -np.inf, dtype=np.float32)
    l_i = ct.full((), 0.0, dtype=np.float32)

    for gg in range(0, BLOCK_G):
        valid_g = gg < NUM_GROUPS

        gm_raw = ct.astype(
            ct.load(
                tmp_m,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.NEG_INF,
            ),
            np.float32,
        )
        gl_raw = ct.astype(
            ct.load(
                tmp_l,
                index=(b, h, gg),
                shape=(),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            np.float32,
        )

        gm = ct.where(valid_g, gm_raw, -np.inf)
        gl = ct.where(valid_g, gl_raw, 0.0)

        m_new = ct.maximum(m_i, gm)
        m_safe = ct.where(m_new == -np.inf, 0.0, m_new)

        alpha = ct.where(l_i > 0.0, ct.exp(m_i - m_safe), 0.0)
        beta = ct.where(gl > 0.0, ct.exp(gm - m_safe), 0.0)

        group_acc = ct.astype(
            ct.load(
                tmp_acc,
                index=(b, h, gg, dchunk),
                shape=(1, 1, 1, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_D,)),
            np.float32,
        )

        acc = acc * alpha + group_acc * beta
        l_i = l_i * alpha + gl * beta
        m_i = m_new

    out = acc / (l_i + 1.0e-10)
    ct.store(output, index=(b, h, dchunk), tile=out.reshape((1, 1, BLOCK_D)))


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim), device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    BLOCK_D = 32
    BLOCK_G = 16
    occupancy = 8

    num_groups = ct.cdiv(num_blocks, BLOCK_K)

    tmp_acc = torch.empty((batch, heads, num_groups, head_dim), device=mid_o.device, dtype=torch.float32)
    tmp_m = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)
    tmp_l = torch.empty((batch, heads, num_groups), device=mid_o.device, dtype=torch.float32)

    block_seq_is_tensor = isinstance(block_seq, torch.Tensor)
    if block_seq_is_tensor:
        block_seq_tensor = block_seq
        block_seq_value = 1
    else:
        block_seq_tensor = b_seqlen
        block_seq_value = int(block_seq)

    stream = torch.cuda.current_stream()

    partial_kernel = _partial_flash_decode_kernel.with_hints(occupancy=occupancy)
    final_kernel = _final_flash_decode_kernel.with_hints(occupancy=occupancy)

    grid_partial = (num_groups, batch * heads, ct.cdiv(head_dim, BLOCK_D))
    ct.launch(
        stream,
        grid_partial,
        partial_kernel,
        (
            mid_o,
            mid_o_lse,
            b_seqlen,
            block_seq_tensor,
            tmp_acc,
            tmp_m,
            tmp_l,
            block_seq_value,
            block_seq_is_tensor,
            num_blocks,
            heads,
            head_dim,
            BLOCK_K,
            BLOCK_D,
        ),
    )

    grid_final = (batch * heads, ct.cdiv(head_dim, BLOCK_D), 1)
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
            "occupancy": occupancy,
            "num_groups": num_groups,
            "block_seq_is_tensor": block_seq_is_tensor,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
