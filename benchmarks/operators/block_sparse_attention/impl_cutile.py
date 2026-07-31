import math
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner


ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(occupancy=16)
_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [4, 8, 16, 32]]


@ct.kernel
def block_sparse_attention_kernel(
    Out, Q, K, V,
    csr_row_indices, csr_col_indices,
    csr_row_stride_h: ConstInt, csr_col_stride_h: ConstInt,
    num_layout: ConstInt, softmax_scale: ct.Constant[float],
    num_heads: ConstInt, num_kv_heads: ConstInt, total_seq_len: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt,
    BLOCK_D: ConstInt, NUM_D_BLOCKS: ConstInt,
):
    start_m = ct.bid(0)
    off_bh = ct.bid(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups


    q = ct.load(
        Q,
        index=(off_b, off_h, start_m, 0),
        shape=(1, 1, BLOCK_M, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((BLOCK_M, BLOCK_D))
    if NUM_D_BLOCKS >= 2:
        q2 = ct.load(
            Q,
            index=(off_b, off_h, start_m, 1),
            shape=(1, 1, BLOCK_M, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_M, BLOCK_D))

    m_i = ct.full((BLOCK_M, 1), -float("inf"), dtype=ct.float32)
    l_i = ct.zeros((BLOCK_M, 1), dtype=ct.float32)
    acc = ct.zeros((BLOCK_M, BLOCK_D), dtype=ct.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = ct.zeros((BLOCK_M, BLOCK_D), dtype=ct.float32)

    layout_h = off_h % num_layout
    row_idx_ptr = layout_h * csr_row_stride_h + start_m
    start_l = ct.load(csr_row_indices, index=(row_idx_ptr,), shape=())
    end_l = ct.load(csr_row_indices, index=(row_idx_ptr + 1,), shape=())

    offs_m = (
        start_m * BLOCK_M
        + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
    )
    valid_m = offs_m < total_seq_len

    l = start_l
    while l < end_l:
        col_idx_ptr = layout_h * csr_col_stride_h + l
        col_idx = ct.load(csr_col_indices, index=(col_idx_ptr,), shape=())
        start_n_tile = col_idx

        k = ct.load(
            K,
            index=(off_b, off_h_kv, start_n_tile, 0),
            shape=(1, 1, BLOCK_N, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_N, BLOCK_D))
        qk = ct.mma(
            q,
            ct.transpose(k, 0, 1),
            ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32),
        )

        if NUM_D_BLOCKS >= 2:
            k2 = ct.load(
                K,
                index=(off_b, off_h_kv, start_n_tile, 1),
                shape=(1, 1, BLOCK_N, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_N, BLOCK_D))
            qk = ct.mma(q2, ct.transpose(k2, 0, 1), qk)

        qk = qk * softmax_scale

        offs_n = (
            start_n_tile * BLOCK_N
            + ct.expand_dims(ct.arange(BLOCK_N, dtype=ct.int32), 0)
        )
        valid_n = offs_n < total_seq_len
        mask = (offs_m >= offs_n) & valid_m & valid_n
        qk = ct.where(mask, qk, -float("inf"))


        row_has_prev = l_i > 0.0
        row_has_valid = (
            ct.sum(ct.astype(mask, ct.int32), axis=1, keepdims=True) > 0
        )
        block_max = ct.where(
            row_has_valid,
            ct.max(qk, axis=1, keepdims=True),
            -float("inf"),
        )
        m_i_new = ct.maximum(m_i, block_max)
        row_has_any = row_has_prev | row_has_valid
        m_safe = ct.where(row_has_any, m_i_new, 0.0)
        alpha = ct.where(row_has_prev, ct.exp(m_i - m_safe), 0.0)
        p = ct.where(mask, ct.exp(qk - m_safe), 0.0)
        l_i_new = l_i * alpha + ct.sum(p, axis=1, keepdims=True)

        acc = acc * alpha
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * alpha

        p_casted = ct.astype(p, Q.dtype)
        v = ct.load(
            V,
            index=(off_b, off_h_kv, start_n_tile, 0),
            shape=(1, 1, BLOCK_N, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_N, BLOCK_D))
        acc = ct.mma(p_casted, v, acc)

        if NUM_D_BLOCKS >= 2:
            v2 = ct.load(
                V,
                index=(off_b, off_h_kv, start_n_tile, 1),
                shape=(1, 1, BLOCK_N, BLOCK_D),
                padding_mode=ct.PaddingMode.ZERO,
            ).reshape((BLOCK_N, BLOCK_D))
            acc2 = ct.mma(p_casted, v2, acc2)

        l_i = l_i_new
        m_i = ct.where(row_has_any, m_i_new, m_i)
        l = l + 1

    l_safe = ct.where(l_i > 0.0, l_i, 1.0)
    out0 = ct.astype(acc / l_safe, Out.dtype).reshape(
        (1, 1, BLOCK_M, BLOCK_D)
    )
    ct.store(Out, index=(off_b, off_h, start_m, 0), tile=out0)

    if NUM_D_BLOCKS >= 2:
        out1 = ct.astype(acc2 / l_safe, Out.dtype).reshape(
            (1, 1, BLOCK_M, BLOCK_D)
        )
        ct.store(Out, index=(off_b, off_h, start_m, 1), tile=out1)


_tuner = CutileAutotuner(block_sparse_attention_kernel)


def run(
    Q, K, V,
    layout_csr_row_indices, layout_csr_col_indices,
    layout_csr_row_stride_h, layout_csr_col_stride_h,
    num_layout, softmax_scale, num_heads, num_kv_heads,
    total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
    block_size: int = None, autotune: bool = False,
):
    batch_size = Q.shape[0]
    D = Q.shape[-1]
    total_d = BLOCK_D * NUM_D_BLOCKS

    assert D == total_d, (
        f"Invalid config: Q.shape[-1]={D}, BLOCK_D={BLOCK_D}, "
        f"NUM_D_BLOCKS={NUM_D_BLOCKS}, expected D={total_d}"
    )
    assert 1 <= NUM_D_BLOCKS <= 2, "Kernel currently supports one or two D blocks"
    assert K.shape[-1] == D and V.shape[-1] == D
    assert total_seq_len == Q.shape[2]
    assert num_heads % num_kv_heads == 0

    out = torch.empty_like(Q)
    grid = (math.ceil(total_seq_len / BLOCK_M), batch_size * num_heads, 1)
    stream = torch.cuda.current_stream()
    args = (
        out, Q, K, V,
        layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, float(softmax_scale),
        num_heads, num_kv_heads, total_seq_len,
        BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS,
    )

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(
                batch_size, num_heads, total_seq_len,
                BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS, str(Q.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: args,
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, args)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
