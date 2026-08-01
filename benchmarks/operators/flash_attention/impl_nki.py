"""NKI flash attention (causal), tiled over 128x128 blocks, two-pass softmax.

head_dim == 128 (config default) exactly matches PMAX, so QK^T and P@V are
each a single-K-tile nc_matmul per (query-block, key-block) pair; Q/K need
an nc_transpose per block to get head_dim onto partitions for QK^T's
contraction (matmul_fp32_fp16_fp8's A-transpose pattern), V doesn't (its
key-position rows are already on partitions, matching P@V's contraction).

This does NOT use the classic single-pass online-softmax recurrence
(rescaling a running accumulator by exp(old_max - new_max) as each
key-block's max is discovered) -- that formulation, even using the same
primitives that work everywhere else in this file, produced large errors
whenever a query block spanned more than one key-block (confirmed via
temp/check_flash4.py: exact for a single key-block, wrong starting at the
second, independent of which primitive -- sequential_range vs plain range,
NEG_INF magnitude, or scratch-tile reuse -- was varied to isolate it).
Root cause not pinned down further given time constraints; sidestepped by
recomputing each key-block's QK^T twice -- once in a max-only pass, once in
a sum/output pass after the row max is already final -- so both
accumulators are plain sums (`acc[...] = nl.add(acc, ...)`), the same
pattern already proven correct in layernorm/kl_divergence/gaussian_blur,
with no cross-iteration rescale anywhere.
"""
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
    def flash_attention_kernel(Q, K, V, causal, scale):
        seq_len, head_dim = Q.shape
        num_blocks = (seq_len + PMAX - 1) // PMAX
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

            kj_limit = (qi + 1) if causal else num_blocks

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

                grid = nl.mgrid[0:PMAX, 0:PMAX]
                row_iota = nisa.iota(expr=grid.p, dtype=nl.int32)
                col_iota = nisa.iota(expr=grid.x, dtype=nl.int32)
                key_col_valid = col_iota < (seq_len - k_offset)
                # One uniform global-position comparison instead of a
                # kj==qi-branching local one: for kj<qi every (row,col) in
                # the block trivially satisfies it (q_offset > k_offset+127
                # >= k_offset+col), and for kj==qi it reduces to the usual
                # local row>=col triangle. A python-level `if causal and
                # kj==qi: ... else: ...` choosing between two *different*
                # tensor expressions per kj iteration produced wrong results
                # for reasons not fully root-caused given time constraints
                # (confirmed via temp/check_flash_noncausal.py: identical
                # kernel with the branch never active, i.e. always taking
                # the same path, is exact) -- this single always-the-same
                # expression sidesteps it.
                q_pos = nl.add(row_iota, q_offset)
                k_pos = nl.add(col_iota, k_offset)
                causal_ok = nl.greater_equal(q_pos, k_pos)
                valid = (key_col_valid & causal_ok) if causal else key_col_valid

                neg_tile = nl.full(scores.shape, NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
                return nl.where(valid, scores, neg_tile), mask_k, k_offset, k_partition

            # Pass 1: row max across all key-blocks (no accumulator rescale).
            running_max = nl.full((PMAX, 1), NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            for kj in range(kj_limit):
                scores_masked, _, _, _ = _scores(kj)
                block_max = nl.max(scores_masked, axis=1, keepdims=True)
                running_max[...] = nl.maximum(running_max, block_max)

            # Pass 2: exp-sum and weighted-V-sum using the now-final max.
            running_sum = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            out_acc = nl.zeros((PMAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            for kj in range(kj_limit):
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


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    batch, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)
    out = torch.empty_like(q)
    for b in range(batch):
        for h in range(n_heads):
            res = flash_attention_kernel(q[b, h].contiguous(), k[b, h].contiguous(),
                                          v[b, h].contiguous(), causal, scale)
            out[b, h] = res
    return out


def get_last_config() -> dict | None:
    return None
