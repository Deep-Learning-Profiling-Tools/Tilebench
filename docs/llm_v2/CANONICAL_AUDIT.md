# CANONICAL_AUDIT — 45 operator contracts (status as recorded in audit.json)

Status values: `draft` = both extractions reconcile (aligned / mapping-only) and the contract text is ready for owner review; `needs-review` = at least one aspect or boundary question needs a human decision; `approved` = set only by the study owner. No operator is approved by this tool.

| operator | status | needs-review aspects | F/Q consistent | human ref comparable | review items |
|---|---|---|---|---|---|
| 1d_conv | draft | - | true | True | 0 |
| 2d_conv | draft | - | true | True | 0 |
| 2d_max_pooling | draft | - | true | True | 0 |
| 3d_conv | draft | - | true | True | 0 |
| argmax | draft | - | true | True | 0 |
| batch_normalization | draft | - | true | True | 0 |
| batched_matmul | needs-review | preprocessing, intermediate_storage | false | False | 3 |
| bitonic_sort | draft | - | unclear | True | 2 |
| block_sparse_attention | draft | - | true | True | 1 |
| cross_entropy | draft | - | true | True | 0 |
| dequantize_rowwise | draft | - | true | True | 0 |
| destindex | needs-review | preprocessing, mutation | unclear | False | 3 |
| dropout | draft | - | true | True | 0 |
| flash_attention | draft | - | unclear | True | 1 |
| flash_decode | draft | - | true | True | 1 |
| fused_activation | draft | - | true | True | 0 |
| gaussian_blur | needs-review | precision | true | True | 2 |
| histogramming | draft | - | false | True | 2 |
| interleave | draft | - | true | True | 0 |
| jacobi_stencil_2d | draft | - | true | True | 0 |
| kl_divergence | draft | - | true | True | 0 |
| l2_norm | draft | - | true | True | 0 |
| layernorm | draft | - | true | True | 0 |
| leaky_relu | draft | - | true | True | 0 |
| linear_self_attention | draft | - | true | True | 0 |
| matmul_fp32_fp16_fp8 | needs-review | preprocessing, intermediate_storage | false | False | 3 |
| matmul_int8 | needs-review | preprocessing, intermediate_storage | false | False | 3 |
| matrix_copy | draft | - | true | True | 0 |
| matrix_transpose | draft | - | true | True | 0 |
| mean_reduction | draft | - | true | True | 1 |
| moe_topk_gating | draft | - | true | True | 1 |
| mul2 | draft | - | true | True | 0 |
| quantize_global | needs-review | - | true | True | 2 |
| radix_sort | needs-review | - | false | True | 2 |
| relu | draft | - | true | True | 0 |
| reverse_array | draft | - | true | True | 0 |
| rmsnorm | draft | - | true | True | 1 |
| rope | draft | - | false | True | 1 |
| sigmoid | draft | - | true | True | 0 |
| softmax | draft | - | true | True | 0 |
| streamk_matmul | needs-review | preprocessing, intermediate_storage | false | False | 3 |
| swiglu | draft | - | true | True | 0 |
| top_k_selection | draft | - | unclear | True | 1 |
| vector_add | draft | - | true | True | 1 |
| weight_dequant | draft | - | true | True | 1 |

Totals: {'draft': 37, 'needs-review': 8, 'approved': 0, 'missing': 0, 'invalid': 0}

## Review items by operator

### batched_matmul (needs-review)
- Decide whether run() may consume B through a pre-transposed/packed layout and, if so, whether that repack must be performed inside every timed call or may be prepared once and cached: the Triton path memoises B^T per tensor identity (impl_triton.py 41-51, 136) so its steady-state latency excludes a transpose the reference never performs, while the cuTile path reads row-major B directly (impl_cutile.py 58-62, 83); the contract currently carries a provisional no-repack, no-cache rule.
- Confirm that bytes_expr = 3 * BATCH * M * M * dtype_size (one read of A and B, one write of C) is the intended boundary; the untimed cached B^T materialisation is uncounted, so the Triton column is not comparable to a contract-conforming generated implementation (human_reference_comparable = false) until the first item is resolved or the Triton implementation is changed.
- Decide whether the identity-keyed B^T cache (impl_triton.py 41-51), which would silently return stale results if B were mutated in place between calls, should be fixed before that implementation serves as a timing reference.
- aspect `preprocessing`: Genuine F-Q boundary divergence: one side benefits from an untimed, cached pre-transpose that the reference (torch.matmul on row-major views, impl_torch.py 9-11) does not need, while the other side and the reference consume B as given. The contract cannot pick either; the question is stated in review_items and the contract carries a provisional no-repack rule.
- aspect `intermediate_storage`: A persistent input-derived buffer that outlives the call exists on one side only; it is the storage side of the preprocessing divergence.
### bitonic_sort (draft)
- Confirm that bytes_expr = 2 * n * dtype_size (compulsory traffic) is the intended figure of merit for a multi-pass in-place sort whose timed work is the pad pass plus L(L+1)/2 passes over M = next_power_of_2(n) elements (impl_triton.py 58-83, impl_cutile.py 74-107); the formula is identical for both manual implementations, so it does not affect their comparison, but it understates executed traffic by a factor that grows with log^2 n.
- Both manual implementations launch one kernel per (k, j) stage with every exchange in global memory; the contract follows the guide's dependency-based rule and permits consecutive stages whose partner distances fit inside one program's slice to be fused into one launch (a register/shared-memory local phase). Decide whether that fusion freedom is acceptable or whether, for parity with the manual implementations, the contract should require one global synchronisation per stage.
### block_sparse_attention (draft)
- Confirm that flops_expr should stay tied to the generator's fixed window density (executed block work, including the masked half of diagonal blocks) rather than being derived from the CSR arrays; it is correct for the benchmark layout and both manual kernels but silently wrong for any other layout.
### destindex (needs-review)
- Output-buffer policy: decide whether a generated run() must (a) allocate the outputs and initialise them from o_nope/o_rope on every call (reference semantics, doubles the timed traffic relative to bytes_expr), or (b) may keep persistent output buffers keyed to the o_* input identity, initialised once and rewritten in place on each call (the manual boundary; relies on module-level tensor-identity state); writing into o_* directly is forbidden on both sides.
- bytes_expr consistency: if policy (a) is chosen, decide whether bytes_expr should include the per-call read of o_* and the corresponding write (n * dtype_size * 4) so that SOL figures reflect the timed work; if (b) is chosen, human_reference_comparable can be set true.
- Define the required result for rows not addressed by dest_loc (they must equal the o_* values) and whether the evaluator should test a non-permutation dest_loc, which the current generator never produces.
- aspect `preprocessing`: Both sides (and impl_torch.py L4-12) agree with each other, but the policy itself is a timed-boundary question: a generated run() that initialises fresh outputs from o_* on every call times an extra read+write of both outputs (doubling the traffic counted by bytes_expr), one that keeps per-identity persistent buffers reproduces the manual boundary but relies on module-level state keyed by tensor identity (flagged as suspicious by the static checker), and one that writes into o_* directly would mutate inputs (forbidden). The human must choose the required policy.
- aspect `mutation`: Agreed: inputs are never modified. Open: whether the returned tensors may be the same objects across calls (rewritten in place) or must be distinct per call; rows not named by dest_loc keep the o_* values copied on the first call.
### flash_attention (draft)
- (non-blocking, metric) flops_expr counts dense attention while causal is fixed to true and both implementations skip the fully masked key tiles (about half the work): decide whether to keep the dense-equivalent convention or switch flops_expr to the causal count (seq_len*(seq_len+1)/2 style) so that TFLOPS and compute-bound SOL figures reflect executed work.
### flash_decode (draft)
- (non-blocking) Both implementations omit the reference's 1e-10 denominator epsilon; results agree within fp32 tolerance for every benchmark input (b_seqlen >= 1 so the denominator is >= 1), but a batch row with b_seqlen <= 0 would give 0/0 = NaN instead of the reference's 0: confirm that rows with zero valid splits are out of scope (the contract currently leaves them unspecified).
### gaussian_blur (needs-review)
- Per-tap product precision for fp16 inputs: one implementation multiplies pixel and weight in the input dtype (fp16 rounding per tap) before fp32 accumulation, the other converts both to fp32 first. Decide whether the contract requires fp32 products (stricter, matches the fp32 reference path more closely) or permits input-dtype products (both currently pass the atol 0.1 / rtol 1e-2 tolerance); the contract text leaves it open.
- (non-blocking, metric) bytes_expr/flops_expr assume square images (input_rows squared); correct for the config sweep but wrong if input_cols != input_rows is ever swept.
- aspect `precision`: Genuine precision-path divergence for fp16 inputs: fp16 products (one extra rounding per tap, relative error about 2^-11 each) versus fp32 products. Both pass the loose verify tolerance (atol 0.1, rtol 1e-2) and both are identical for fp32 inputs. The contract must not pick a side: the human has to decide whether per-tap products must be formed in fp32.
### histogramming (draft)
- (non-blocking, metric) bytes_expr omits the per-call scratch zero-fill (num_partials * num_bins * 4 B memset inside run()), the N atomic read-modify-writes into the scratch and the reduce pass's full scratch read, all of which are inside the timed window on both sides: decide whether to keep the compulsory-traffic convention (input read + output write) or to add the scratch traffic so that bandwidth/SOL figures are not understated in the small-N / large-num_bins corner.
- (non-blocking, scope) Both implementations count with global-memory atomic adds into a private row of a zero-filled global scratch; the contract requires the two-level structure (private partial histograms, then a cross-partial sum) and a scratch initialised inside run(), but leaves the private-row update mechanism (per-element atomics versus on-chip pre-aggregation such as a register/shared-memory histogram primitive followed by additive updates of the row) as a permitted mapping. Confirm that this freedom is intended, since on-chip pre-aggregation can cut the scratch traffic the manual kernels pay.
### matmul_fp32_fp16_fp8 (needs-review)
- Decide whether the K-major repack of B is inside the timed algorithm: the contract requires any repack to be per call and uncached, but the manual Triton timing excludes it (bt cached in _bt_cache, impl_triton.py:32-39,132); either re-measure the Triton reference with a per-call transpose or direct (K, N) consumption, or mark its numbers as not comparable to contract-conforming implementations.
- Fix or flag the manual Triton _bt_cache (impl_triton.py:32-39): a plain dict keyed by (data_ptr, shape, dtype) that is never evicted is a stale-address correctness hazard when a freed B's address is reused and retains every (N, K) copy across the 60-case sweep; matmul_int8 already uses a WeakTensorKeyDictionary for the same purpose.
- Confirm the fp32 reference precision at verification time: impl_torch.py:4 sets torch.backends.cuda.matmul.allow_tf32 = True at import and never restores it, while matmul_int8's reference sets it to False inside run() without restoring; the contract permits TF32 operand rounding on the assumption that the reference is also TF32-class under the atol 5.0 / rtol 0.1 tolerance.
- aspect `preprocessing`: Genuine F/Q boundary divergence: one side's steady-state timed region is 'GEMM with a prepacked K-major B', the other's is 'GEMM over the delivered layout'. The reference applies a similar cached repack only for fp8 (impl_torch.py:10-18). Not resolvable by intersection; the contract requires any repack to be per call and uncached and lists the question as open.
- aspect `intermediate_storage`: Cross-call persistent operand copy vs none; same root cause as the preprocessing divergence. The address-keyed plain dict is also a stale-address hazard (a freed B's address reused by a same-shape, same-dtype B would receive stale data).
### matmul_int8 (needs-review)
- Decide whether the K-major repack of the packed B is inside the timed algorithm: the contract requires any repack to be per call and uncached, but the manual Triton timing excludes it (bt cached in a WeakTensorKeyDictionary, impl_triton.py:16-24,128); either re-measure the Triton reference with a per-call transpose or direct (K_b, N) consumption, or mark its numbers as not comparable to contract-conforming implementations.
- Decide whether the contract must require correctness for arbitrary K_b: the manual Triton kernel enforces K % (4*BLOCK_K) == 0 by static_assert (impl_triton.py:46-49) and the manual cuTile slab index k_a = i*num_tiles_kb + j (impl_cutile.py:57) is only exact when K_b is a multiple of its K tile, while a zero-padded packed byte decodes to -1; the contract currently requires correctness for the swept shapes and masking of any tail.
- impl_torch.py:17 sets torch.backends.cuda.matmul.allow_tf32 = False process-wide inside run() and never restores it; decide whether the harness isolates this, since it changes the fp32 reference of matmul_fp32_fp16_fp8 (which sets the flag True at import) when both run in one process.
- aspect `preprocessing`: Genuine F/Q boundary divergence (same shape as matmul_fp32_fp16_fp8, but without the stale-address hazard because the key is weak). The contract requires any repack to be per call and uncached and lists the question as open.
- aspect `intermediate_storage`: Cross-call persistent operand copy vs none; same root cause as the preprocessing divergence.
### mean_reduction (draft)
- Non-blocking metric accuracy: bytes_expr 'M * N * dtype_size' omits the M * 4-byte float32 output write that every implementation performs; decide whether to change it to 'M * N * dtype_size + M * 4' (at most about 0.4 percent over the sweep; does not affect the contract text).
### moe_topk_gating (draft)
- Non-blocking: the tie-breaking rule among equal logits is unspecified by the reference (torch.topk) and by both implementations; the generator guarantees distinct values per row so it is never exercised. Decide whether the contract should mandate a rule (for example lowest column index first among equal values) or keep the current wording that ties are outside the verified input domain.
### quantize_global (needs-review)
- BLOCKING (study owner): the operator is named quantize_global but at this SHA the reference and both implementations are a plain float32 -> float16 cast with no absmax reduction, no scale factor and no integer output; decide whether to keep the operator as this cast (and rename or document it) or to redefine it as a scaled quantisation, which would require a new reference, new implementations, a new generator and a new contract. The contract below describes the cast only and must not be approved until this is settled.
- Non-blocking: the generator ignores its dtype argument and always produces float32 (tensors.py 175-176) and config.yaml pins dtype to fp32; confirm the single-dtype sweep is intended, since bytes_expr's dtype_size term is only meaningful for fp32 input.
### radix_sort (needs-review)
- BLOCKING (metric vs contract): bytes_expr hard-codes 16 passes (the 2-bit-digit mapping) while the contract leaves digit width and pass count free; decide either to pin the pass count in the contract (making the formula a faithful traffic model but constraining the mapping) or to keep the freedom and re-express or annotate the metric (for example report bytes per actual pass, or treat the formula as a fixed nominal '16-pass-equivalent' figure that is only comparable between implementations with the same pass count).
- Non-blocking: the working-copy clone inside run() (2 * n * dtype_size bytes) and the histogram/scan scratch traffic are not counted by bytes_expr; decide whether to add the clone term, which every faithful implementation performs.
### rmsnorm (draft)
- bytes_expr = 3n matches a two-sweep kernel that re-reads x from memory; an implementation that keeps the row on chip moves ~2n and will report above the 3n model. Decide whether the metric stays at 3n (current, matches both references) or is changed to the minimal 2n bound. Non-blocking: the contract permits either mapping.
### rope (draft)
- Both references clone q inside the timed run() and rotate the clone in place, moving ~2x the bytes_expr model; the contract requires a non-aliasing fresh output but permits an out-of-place mapping that meets the 2n model. Decide whether bytes_expr stays as the minimal 2n bound (generated out-of-place kernels may then outperform the references for non-algorithmic reasons) or whether the references should be re-measured out-of-place. Non-blocking for the contract text.
### streamk_matmul (needs-review)
- Prepacked-operand divergence: the Triton reference loads a host-transposed (N, K) copy of B cached per tensor identity (impl_triton.py 63-71, 264), untimed after the first call, while the cuTile reference consumes B as the given (K, N) tensor. Decide whether a generated implementation may prepare such an operand copy outside the timed region (then the cuTile reference is disadvantaged and the contract's preprocessing rule must change) or must do everything in-run (then the Triton reference must be re-measured with the transpose inside run() or rewritten to consume B directly). The contract currently requires the in-run, uncached behaviour; human_reference_comparable is false until this is resolved.
- bytes_expr omits traffic that both references pay inside run(): the per-call zero-fill of the fp32 accumulation buffer, the atomic read-modify-write of Stream-K tiles and, for fp16/bf16, the fp32 scratch round trip of the final cast. Decide whether the bandwidth model should count these (it affects every implementation of this family equally) or stay the minimal GEMM bound with flops as the primary metric.
- Both references combine Stream-K segments with fp32 atomic adds, so split tiles are not bitwise reproducible run to run and verification relies on atol 1.0 / rtol 1e-2; confirm that this non-determinism is an accepted property of the canonical family (the contract states it) rather than something a generated implementation must avoid.
- aspect `preprocessing`: Prepacked-operand divergence: one side feeds its loads from a host-transposed, identity-cached B^T that the timed region does not pay for after the first call; the other consumes B as given. The contract conservatively requires in-run, uncached handling pending the review decision.
- aspect `intermediate_storage`: The per-call fp32 accumulation buffer is aligned; the cross-call (N, K) operand copy exists on one side only and is the same divergence as the preprocessing item.
### top_k_selection (draft)
- bytes_expr = (N + k) * dtype_size ignores the candidate buffers that the canonical hierarchy writes and re-reads inside run() (about N*K2/B at level 0, up to N/2 bytes-equivalent when B = 2*K2); both references pay it equally. Decide whether the bandwidth model stays the minimal bound (current) or counts level traffic, which would make the metric depend on the chosen block width. Non-blocking for the contract text.
### vector_add (draft)
- The cuTile reference issues its loads and stores without a padding mode (impl_cutile.py 24-26) and is only correct because every swept n is a multiple of its tile; the contract requires correct bounds handling for arbitrary n. Decide whether the reference should be hardened to match (non-blocking: no swept case is affected).
### weight_dequant (draft)
- bytes_expr and flops_expr hard-code N == M (M*M terms) while run() and the generator accept a distinct N; decide whether to generalise the expressions to M*N (no measured case is affected). Non-blocking for the contract text.
