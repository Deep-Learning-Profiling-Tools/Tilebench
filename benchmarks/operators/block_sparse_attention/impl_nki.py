"""NKI block-sparse attention.

Reuses flash_attention/impl_nki.py's tiled causal-attention structure
(same 128x128-block QK^T / P@V via nc_transpose+nc_matmul, two-pass
softmax with no cross-iteration accumulator rescale -- see that file's
docstring for why). The CSR block-sparse layout plus the causal triangle are
decoded into one dense (seq_len, seq_len) 0/1 mask on the host, exactly
mirroring impl_torch.py's own mask construction (it also builds a dense
mask from the CSR layout before running dense attention -- this isn't a
shortcut relative to the reference, it's the same approach). The mask is
zero-padded up to a whole number of 128-blocks, so the kernel needs no
separate in-bounds check: an out-of-range query/key position is already
invalid in the padded mask.
"""
import math
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def block_sparse_kernel(Q, K, V, mask, scale):
        seq_len, head_dim = Q.shape
        num_blocks = mask.shape[0] // PMAX
        NEG_INF = -3.0e38

        out = nl.ndarray((seq_len, head_dim), dtype=Q.dtype, buffer=nl.hbm)

        for qi in range(num_blocks):
            q_offset = qi * PMAX
            q_partition = nl.arange(PMAX)[:, None]
            free_index = nl.arange(head_dim)[None, :]
            mask_q = q_partition < (seq_len - q_offset)

            q_raw = nl.load(Q[q_offset + q_partition, free_index], mask=mask_q, dtype=nl.float32)
            zero_hd = nl.zeros(q_raw.shape, dtype=nl.float32, buffer=nl.sbuf)
            q_tile = nl.where(mask_q, q_raw, zero_hd)

            def _scores(kj):
                q_t = nisa.nc_transpose(q_tile)
                k_offset = kj * PMAX
                k_partition = nl.arange(PMAX)[:, None]
                mask_k = k_partition < (seq_len - k_offset)
                zero_k = nl.zeros((PMAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                k_raw = nl.load(K[k_offset + k_partition, free_index], mask=mask_k, dtype=nl.float32)
                k_tile = nl.where(mask_k, k_raw, zero_k)
                k_t = nisa.nc_transpose(k_tile)

                scores_psum = nisa.nc_matmul(q_t, k_t)
                scores = nl.multiply(nl.copy(scores_psum, dtype=nl.float32), scale)

                mask_free = nl.arange(PMAX)[None, :]
                mask_tile = nl.load(mask[q_offset + q_partition, k_offset + mask_free], dtype=nl.float32)
                valid = nl.greater(mask_tile, 0.5)

                neg_tile = nl.full(scores.shape, NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
                return nl.where(valid, scores, neg_tile), mask_k, k_offset, k_partition

            running_max = nl.full((PMAX, 1), NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            for kj in range(num_blocks):
                scores_masked, _, _, _ = _scores(kj)
                block_max = nl.max(scores_masked, axis=1, keepdims=True)
                running_max[...] = nl.maximum(running_max, block_max)

            running_sum = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            out_acc = nl.zeros((PMAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            for kj in range(num_blocks):
                scores_masked, mask_k, k_offset, k_partition = _scores(kj)
                p = nl.exp(nl.subtract(scores_masked, running_max))
                running_sum[...] = nl.add(running_sum, nl.sum(p, axis=1, keepdims=True))

                zero_v = nl.zeros((PMAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                v_raw = nl.load(V[k_offset + k_partition, free_index], mask=mask_k, dtype=nl.float32)
                v_tile = nl.where(mask_k, v_raw, zero_v)

                p_t = nisa.nc_transpose(p)
                pv_psum = nisa.nc_matmul(p_t, v_tile)
                out_acc[...] = nl.add(out_acc, nl.copy(pv_psum, dtype=nl.float32))

            final_out = nl.divide(out_acc, running_sum)
            final_cast = nl.add(final_out, 0.0, dtype=Q.dtype)
            nl.store(out[q_offset + q_partition, free_index], value=final_cast, mask=mask_q)

        return out


def _build_dense_mask(H, M, N, layout_csr_row_indices, layout_csr_col_indices,
                       layout_csr_row_stride_h, layout_csr_col_stride_h,
                       num_layout, BLOCK_M, BLOCK_N, device):
    """Mirrors impl_torch.py's CSR decode + causal-triangle combination."""
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(N / BLOCK_N)

    sparse_mask = torch.zeros((H, M, N), dtype=torch.bool, device=device)
    for h in range(H):
        layout_h = h % num_layout
        for r in range(num_rows):
            start_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r].item()
            end_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r + 1].item()
            for l in range(start_l, end_l):
                c = layout_csr_col_indices[layout_h * layout_csr_col_stride_h + l].item()
                r_start, r_end = r * BLOCK_M, min((r + 1) * BLOCK_M, M)
                c_start, c_end = c * BLOCK_N, min((c + 1) * BLOCK_N, N)
                sparse_mask[h, r_start:r_end, c_start:c_end] = True

    causal_mask = torch.tril(torch.ones(M, N, dtype=torch.bool, device=device))
    return sparse_mask & causal_mask.unsqueeze(0)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False, **kwargs):
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape
    head_groups = H // H_kv

    dense_mask = _build_dense_mask(H, M, N, layout_csr_row_indices, layout_csr_col_indices,
                                    layout_csr_row_stride_h, layout_csr_col_stride_h,
                                    num_layout, BLOCK_M, BLOCK_N, Q.device)

    Mp = ((M + PMAX - 1) // PMAX) * PMAX
    Np = ((N + PMAX - 1) // PMAX) * PMAX
    mask_padded = torch.zeros(H, Mp, Np, dtype=torch.float32, device=Q.device)
    mask_padded[:, :M, :N] = dense_mask.to(torch.float32)

    out = torch.empty(B, H, M, D, dtype=Q.dtype, device=Q.device)
    for b in range(B):
        for h in range(H):
            kv_h = h // head_groups
            res = block_sparse_kernel(Q[b, h].contiguous(), K[b, kv_h].contiguous(),
                                       V[b, kv_h].contiguous(), mask_padded[h], softmax_scale)
            out[b, h] = res
    return out


def get_last_config() -> dict | None:
    return None
