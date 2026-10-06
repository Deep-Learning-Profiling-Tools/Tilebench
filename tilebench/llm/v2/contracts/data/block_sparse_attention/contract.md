# block_sparse_attention: canonical algorithm contract

## Functional semantics

Causal block-sparse scaled-dot-product attention with grouped-query heads.
For batch `b`, query head `h` and query position `m`, with kv head
`h_kv = h // (num_heads // num_kv_heads)` and layout head
`lh = h % num_layout`:

```
out[b, h, m, :] = sum over n in A(b, h, m) of
    softmax_n( softmax_scale * Q[b, h, m, :] . K[b, h_kv, n, :] ) * V[b, h_kv, n, :]
```

where the admissible key set `A` contains every `n < total_seq_len` with
`n <= m` (causal) such that the block `(m // BLOCK_M, n // BLOCK_N)` is
listed in the CSR layout of `lh`. The softmax is taken over `A` only. A
query row with no admissible key produces zeros. This is what the reference
computes with a block mask built from the CSR layout ANDed with
`q_idx >= kv_idx`, `scale=softmax_scale` and grouped-query attention
enabled.

The entry point is called as `run(Q, K, V, layout_csr_row_indices,
layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h,
num_layout, softmax_scale, num_heads, num_kv_heads, total_seq_len,
BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS)`; all eighteen inputs
are positional. The entry point is called positionally with exactly the inputs listed below; no keyword arguments are passed.

## Inputs and outputs

- `Q`: `(B, H, M, D)`, contiguous, fp16. `K`, `V`: `(B, H_kv, M, D)`,
  contiguous, fp16. All read-only. `H` is a multiple of `H_kv`.
- `layout_csr_row_indices`: int32, CSR row pointers per layout head, head
  `lh` starting at `lh * layout_csr_row_stride_h`; row block `r` of head
  `lh` lists its key blocks at positions `[ptr[r], ptr[r+1])`.
- `layout_csr_col_indices`: int32, key-block indices per layout head, head
  `lh` starting at `lh * layout_csr_col_stride_h`; entries beyond the last
  listed block are zero padding and must not be visited.
- `num_layout`, `num_heads`, `num_kv_heads`, `total_seq_len` (`== M`):
  ints. `softmax_scale`: float.
- `BLOCK_M`, `BLOCK_N`: the query-block and key-block granularity of the
  CSR layout. They are problem parameters supplied with the inputs, not
  tunables: the layout is only meaningful in these units.
- `BLOCK_D`, `NUM_D_BLOCKS`: a decomposition of the head dimension,
  `D == BLOCK_D * NUM_D_BLOCKS` (`NUM_D_BLOCKS` is one or two). They are
  inputs; whether the head dimension is processed as one tile or as
  `NUM_D_BLOCKS` chunks with separate accumulators is a free mapping
  choice (the results differ only in summation order).
- `EVEN_M`, `EVEN_N`: booleans stating whether `M` is a multiple of
  `BLOCK_M` / `BLOCK_N`; they may be ignored (bounds masking is required
  wherever the task's fixed `M` is not a multiple of the chosen block).
- Output: `(B, H, M, D)`, dtype of `Q`, freshly allocated inside `run()` on
  every call, returned as a single tensor. It must not alias any input.

## Required logical stages

1. **Allocate** the output (uninitialised allocation suffices).
2. **Block-sparse online-softmax attention**: one logical pass over the
   listed key blocks. The natural mapping is one program per (query row
   block, batch, head). A program loads its query block, reads the CSR row
   range of its row block for its layout head, and visits each listed key
   block once (stored order is natural; the order is free); for each
   visited block it forms the scaled scores, applies the causal and
   sequence-bound mask, performs the online-softmax update of the running
   row maximum and row sum, rescales the output accumulator, and
   accumulates `P . V`; after the last block it divides by the row sum and
   stores the output block.

Stage 2 is the only device work. No pass may decode the CSR layout into a
dense mask, materialise scores or probabilities in global memory, or split
the key range across programs with a later combine, because each adds a
global intermediate that the canonical algorithm keeps on chip or does not
have. How the work is distributed over launches is not fixed. Other
program-to-work mappings are permitted only if each query row still visits
exactly the key blocks listed for its row block, once each, because
visiting other blocks changes the executed work that defines block-sparse
attention.

## Algorithm family and structure

Flash-attention-style online softmax restricted to a CSR block layout:

- Each key block listed in the CSR column list is visited once; the
  visiting order (stored order is natural) is free.
- Per visited block: `S = softmax_scale * (Q_blk . K_blk^T)` in fp32;
  entries with key index `> query index`, key index `>= total_seq_len` or
  query index `>= total_seq_len` are set to `-inf` (the sequence-bound
  terms matter only where the task's fixed `M` is not a multiple of the
  block size); the mask may be applied to every visited block or only to
  blocks that intersect the diagonal or the sequence end.
- `m_new = max(m_old, rowmax(S))`; a guard (for example clamping the
  maximum at a large negative finite value) must prevent `exp(-inf - -inf)`
  on fully masked rows; `alpha = exp(m_old - m_new)`; `P = exp(S - m_new)`;
  `l = l * alpha + rowsum(P)`; `acc = acc * alpha + P . V`. Each block
  subtracts the running maximum known at that point, with this rescaling;
  only the final normalised result must equal the exact stable softmax
  over `A`. If all listed key blocks of a row block are held on chip at
  once, the maximum and sum may be evaluated over the held values, with no
  streaming loop and no `-inf` initial state (the fully-masked-row guard
  and the zero-sum rule below still apply).
- Final: `out = acc / l` with `l` replaced by one where it is zero, so rows
  with no admissible key yield zeros rather than NaN.
- The natural-exponential formulation with the scale applied to `S` is the
  reference form; an `exp2` formulation with the scale and `log2(e)` folded
  into `Q` or `S` is an equivalent mapping.
- Grouped-query mapping `h_kv = h // (num_heads // num_kv_heads)` and layout
  mapping `h % num_layout` are required.

## Precision and accumulation

- `S`, the running maximum and sum, `alpha`, `P` and the output accumulator
  are fp32.
- `P` is cast to the input dtype (fp16) before the `P . V` product; the
  `Q . K^T` and `P . V` products are tensor-core products with fp32
  accumulation.
- The output is cast to the dtype of `Q` at the store.
- `-inf` is the masking value for scores; out-of-range `Q`/`K`/`V` rows in
  the last block are zero-filled (they are masked out of the softmax
  anyway).

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

`run()` must not decode the CSR arrays on the host, must not build a dense
mask, must not copy, cast or transpose `Q`/`K`/`V`, and must not cache
anything (layout-derived or otherwise) across calls. Besides the launch, it
may assert shape consistency, allocate the output, build host-side load
descriptors or metadata and convert `softmax_scale` to a float. The metric
formulas count one read of `Q`, `K` and `V` and one write of the output;
re-reads of a key block by several query blocks and by the query heads
sharing a kv head are expected and not counted. The flops figure assumes the
benchmark's layout and counts every visited block in full, including the
masked half of diagonal blocks.

## Permitted implementation mappings

- Descriptor-based block loads or pointer-based loads; how the `K` tile is
  transposed for the score product; pipelining of `K`/`V` loads across the
  data-dependent loop.
- Processing `D` as one tile or as `NUM_D_BLOCKS` chunks.
- `exp` versus `exp2` formulation; where the scale is folded.
- Specialising the mask for fully visible versus diagonal blocks, as long
  as the result is identical.
- The order in which a row block's listed key blocks are visited, and
  holding all of them on chip at once (see Algorithm family).
- Passing the scalar inputs as runtime values or compile-time constants.

## Forbidden substitutions

- `flex_attention`, `create_block_mask`,
  `torch.nn.functional.scaled_dot_product_attention`, `torch.softmax`,
  `torch.matmul` / `torch.bmm` / `torch.einsum`, or any library attention
  or GEMM.
- Dense attention over all keys followed by masking; decoding the CSR
  layout into a dense boolean mask; materialising `S` or `P` in global
  memory.
- A two-pass softmax that traverses the listed key blocks twice (separate
  max/sum pass then a normalisation pass), split-key schemes with a
  combine pass, or atomic accumulation: each adds a second logical pass or
  a global intermediate.
- Softmax statistics or accumulators in less than fp32; omitting the
  causal mask, or the sequence-bound mask where the task's fixed `M` is
  not a multiple of the block size; visiting unlisted blocks or the zero
  padding of the column list.
- Writing any input; returning a view of any input.

## Permitted PyTorch operations

- `torch.empty_like(Q)` or `torch.empty((B, H, M, D), dtype=Q.dtype,
  device=Q.device)` for the output.
- `Q.contiguous()` / `K.contiguous()` / `V.contiguous()` only as no-op
  guards on the already-contiguous inputs.
- Reading dtype, shape and device metadata, and obtaining the current
  stream.

Everything else in `torch` is forbidden inside `run()`.
