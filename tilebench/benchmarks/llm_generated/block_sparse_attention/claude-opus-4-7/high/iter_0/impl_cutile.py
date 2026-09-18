import torch

_LAST_CFG: dict = {}


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    raise NotImplementedError(
        "cuTile does not natively support runtime-bounded for-loops needed "
        "for CSR-driven block-sparse attention iteration."
    )


def get_last_config() -> dict | None:
    return None
