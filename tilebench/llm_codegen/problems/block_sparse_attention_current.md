# Block-Sparse Causal Attention (with GQA)

Implement a program that performs scaled dot-product attention restricted to a block-sparse + causal mask, while supporting Grouped Query Attention (the number of KV heads can be smaller than the number of Q heads). The sparse mask is provided in CSR format and selects which (Q-block, K-block) pairs are computed; all other pairs are treated as $-\infty$.

The input consists of:

- `Q`: A 4D array of shape $(B, H, M, D)$, the queries. `H` is the number of Q heads.
- `K`: A 4D array of shape $(B, H_{kv}, M, D)$, the keys. `H_kv` divides `H`.
- `V`: A 4D array of shape $(B, H_{kv}, M, D)$, the values.
- `layout_csr_row_indices`: A 1D int32 array. For each layout-head $l$ and row-block $r$, `layout_csr_row_indices[l * csr_row_stride_h + r]` gives the start offset into `layout_csr_col_indices` for that block-row.
- `layout_csr_col_indices`: A 1D int32 array. The K column-block indices for each row-block, contiguous per row.
- `csr_row_stride_h`, `csr_col_stride_h`, `num_layout`: integers describing the CSR layout strides.
- `softmax_scale`: scalar (typically $1 / \sqrt{D}$).
- `num_heads`, `num_kv_heads`, `total_seq_len`: integers.
- `BLOCK_M`, `BLOCK_N`, `BLOCK_D`, `NUM_D_BLOCKS`, `EVEN_M`, `EVEN_N`: block sizes / shape flags.

The output should be written to the `out` array of shape $(B, H, M, D)$, with the same dtype as Q.

Logic:

1. **GQA mapping**: for Q head $h$, the corresponding KV head index is $h_{kv} = h \,/\, (H / H_{kv})$.
2. **Sparse iteration**: for row block $r$ of head $h$ in layout $l = h \bmod num\_layout$, iterate over the CSR-listed column blocks $c \in$ `csr_col_indices[csr_row_indices[r] : csr_row_indices[r+1]]`. For each $(r, c)$ Q×K block pair, compute the scaled dot products $Q_r K_c^\top \cdot scale$.
3. **Causal mask within each block**: an entry at absolute position $(i, j)$ is masked out if $i < j$ (only positions $j \leq i$ contribute).
4. **Online softmax over the unmasked entries**, accumulating into the output tile.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `out`
- The kernel **must skip masked column blocks** (do not iterate over all blocks and mask-out — iterate only over the CSR-listed blocks)
- Implement online softmax (running max + denominator) per Q-block
- Support GQA via the `H` / `H_kv` mapping
- Tensor cores for the QK and PV matmuls

## Sparse pattern note

The reference setup uses a causal + local-window pattern: each row-block attends to itself and the previous `window_blocks = 2` row-blocks (so up to 3 blocks total, intersected with causal). Skipped blocks contribute nothing.

## Constraints

- $B \geq 1$, $H \geq 1$, $H_{kv} \geq 1$, $M \geq 1$, $D \geq 1$
- $H$ is divisible by $H_{kv}$
- $D$ is divisible by `BLOCK_D`
- `BLOCK_M`, `BLOCK_N`, `BLOCK_D` are powers of two
