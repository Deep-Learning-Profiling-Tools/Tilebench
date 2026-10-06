# Contract revision 2 (2026-10-06): wording corrections, no new algorithm restrictions

Applies NEXT_STEP_EMPIRICAL_CALIBRATION.md §6 to all 45 contracts (`contract_revision` 1 -> 2 in every
`evaluator_rules.json` and `audit.json`; `audit.json` evidence otherwise untouched). Four classes:

1. **aligned ≠ mandatory**: restrictions shared by the two human kernels only by implementation coincidence
   (tile/block sizes, programs per row, grid/raster order, vectorisation, pipelining, launch count without a new
   global-memory intermediate or extra pass) moved to *Permitted implementation mappings*; algorithm-family
   invariants kept and now state their reason (`1-relaxed-M` / `1-kept-A-with-reason` below).
2. **online-softmax degeneration** (attention, softmax, cross_entropy, moe_topk_gating): a row/block held on chip
   may evaluate max/sum directly; intermediate chunks may subtract the running max; only the final normalisation
   must equal the exact stable result.
3. **logical traversals, not physical HBM counts**.
4. **timing boundary**: every *Preprocessing and timing boundary* section now starts with the same paragraph:
   the measured quantity is the GPU time of all device work `run()` causes; host allocation, metadata reads and
   Python are not part of it. `evaluator_rules.json: timing_boundary.notes` says the same.
5. **fixed shape**: 'must not assume tile-multiple shapes / any M,N,K' leftovers replaced by edge handling wherever
   the task's fixed shape is not a multiple of the chosen tile; other shapes need not be supported.

softmax was revised by hand (all five classes); the other 44 by four reviewers working from the audit evidence
and the human kernels at S_main, each logging every change (scratch logs merged in this table).

| operator | relaxed (M) | kept (A, reason added) | class 2 | class 3 | class 4 | shape | rules |
|---|---|---|---|---|---|---|---|
| 1d_conv | 2 | 4 | 0 | 0 | 1 | 0 | 2 |
| 2d_conv | 2 | 4 | 0 | 0 | 1 | 1 | 2 |
| 2d_max_pooling | 3 | 2 | 0 | 0 | 1 | 2 | 2 |
| 3d_conv | 2 | 4 | 0 | 0 | 1 | 1 | 2 |
| argmax | 4 | 1 | 0 | 3 | 1 | 1 | 3 |
| batch_normalization | 2 | 3 | 0 | 4 | 1 | 0 | 3 |
| batched_matmul | 2 | 1 | 0 | 0 | 1 | 2 | 2 |
| bitonic_sort | 3 | 0 | 0 | 0 | 1 | 0 | 2 |
| block_sparse_attention | 5 | 2 | 1 | 1 | 1 | 2 | 2 |
| cross_entropy | 2 | 0 | 2 | 1 | 1 | 0 | 3 |
| dequantize_rowwise | 2 | 1 | 0 | 1 | 1 | 0 | 2 |
| destindex | 0 | 1 | 0 | 2 | 1 | 0 | 2 |
| dropout | 2 | 2 | 0 | 0 | 1 | 1 | 1 |
| flash_attention | 6 | 3 | 4 | 1 | 1 | 0 | 4 |
| flash_decode | 6 | 1 | 1 | 0 | 1 | 0 | 4 |
| fused_activation | 2 | 3 | 0 | 0 | 1 | 1 | 2 |
| gaussian_blur | 3 | 3 | 0 | 0 | 1 | 0 | 2 |
| histogramming | 0 | 1 | 0 | 1 | 1 | 1 | 3 |
| interleave | 2 | 0 | 0 | 3 | 2 | 0 | 2 |
| jacobi_stencil_2d | 2 | 2 | 0 | 0 | 1 | 1 | 2 |
| kl_divergence | 0 | 0 | 0 | 2 | 1 | 0 | 1 |
| l2_norm | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| layernorm | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| leaky_relu | 2 | 1 | 0 | 1 | 1 | 1 | 2 |
| linear_self_attention | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| matmul_fp32_fp16_fp8 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| matmul_int8 | 0 | 1 | 0 | 2 | 1 | 1 | 1 |
| matrix_copy | 2 | 1 | 0 | 3 | 1 | 1 | 2 |
| matrix_transpose | 2 | 1 | 0 | 2 | 1 | 1 | 2 |
| mean_reduction | 1 | 1 | 0 | 4 | 1 | 1 | 4 |
| moe_topk_gating | 2 | 1 | 0 | 2 | 1 | 0 | 3 |
| mul2 | 2 | 1 | 0 | 2 | 1 | 1 | 2 |
| quantize_global | 2 | 1 | 0 | 3 | 1 | 1 | 2 |
| radix_sort | 0 | 1 | 0 | 0 | 3 | 1 | 2 |
| relu | 2 | 1 | 0 | 2 | 1 | 1 | 2 |
| reverse_array | 2 | 1 | 0 | 2 | 1 | 1 | 2 |
| rmsnorm | 2 | 3 | 0 | 0 | 1 | 0 | 2 |
| rope | 2 | 0 | 0 | 0 | 2 | 0 | 2 |
| sigmoid | 3 | 2 | 0 | 2 | 1 | 2 | 2 |
| streamk_matmul | 3 | 2 | 0 | 0 | 1 | 0 | 2 |
| swiglu | 3 | 2 | 0 | 2 | 1 | 2 | 2 |
| top_k_selection | 5 | 1 | 0 | 0 | 2 | 0 | 3 |
| vector_add | 3 | 2 | 0 | 2 | 1 | 2 | 3 |
| weight_dequant | 2 | 1 | 0 | 2 | 1 | 2 | 3 |
| softmax (manual) | 3 | 2 | 2 | 1 | 1 | 1 | 3 |
| **total (44 logged)** | 93 | 63 | 8 | 50 | 49 | 32 | 95 |

## Items the reviewers flagged for the owner (not decided here)

| operator | item | reviewer's action | decision needed |
|---|---|---|---|
| interleave | 'must not be split into separate passes over A and B' relaxed to a mapping | relaxed (each input/output element touched once logically, no intermediate) | revert if the one-pass zip is the algorithm itself |
| histogramming | 'P bounded independently of N' -> 'P is a configuration choice for the fixed N' | relaxed | confirm |
| rmsnorm vs layernorm/l2_norm | rmsnorm forbids passing the row statistic through global memory; layernorm/l2_norm allow split launches with a statistics buffer | left inconsistent (pre-existing) | unify or justify |
| kl_divergence vs mean_reduction | kl_divergence allows a split-row combine, mean_reduction forbids it | left inconsistent (pre-existing) | unify or justify |
| bitonic_sort | 'in place, no second buffer' kept; 'comparisons are strict' kept | kept (probably M) | confirm |
| matmul_int8 | 'each packed byte read once per output tile' kept as A, but mappings allow fields-outer nesting (reloads 4x) | contradiction left | decide A or M |
| matrix_transpose | 'rectangular logical tiles' kept (non-binding: 1-row tiles allowed) | kept | confirm |
| matrix_copy | 'Nothing may be fused with host-side work' kept verbatim | kept | clarify meaning |
| top_k_selection | 'B a power of two', 'contiguous blocks' relaxed; K' = next pow2 >= k kept as A | relaxed/kept | confirm |
| attention ops / flash_decode / argmax | key-tile / split / chunk visiting order freed (fp32 rounding only); flash_attention notes the exp(-inf-(-inf)) guard | relaxed | confirm |
| gaussian_blur | Functional semantics 'must use the kernel_rows/... it is given' could read as any-shape | unchanged | clarify |
| kl_divergence | garbled sentence in Precision ('The intrinsic within the tolerance is acceptable)') | unchanged (protected section) | fix wording |
| 2d_conv, 3d_conv, 2d_max_pooling, jacobi_stencil_2d, batched_matmul, quantize_global, weight_dequant | any-shape sentences inside *Inputs and outputs* rewritten to the fixed-shape rule | edited (protected section) | confirm |
| relu, mul2, matrix_copy, matrix_transpose, reverse_array, sigmoid, block_sparse_attention, radix_sort, mean_reduction | same, applied by the integrator | edited | confirm |

The eight `needs-review` operators keep their *Open review items* and provisional rules unchanged (C1).
Full per-change log (before/after quotes, reasons): `docs/llm_v2/contract_revision_2_changes.json`.
