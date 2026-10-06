# flash_attention: canonical algorithm contract

## Functional semantics
Causal scaled-dot-product attention forward pass. Given `q`, `k`, `v` of
shape (batch, heads, seq_len, head_dim), for every (batch b, head h) and
query row i:

    S[i, j] = (q[b,h,i,:] . k[b,h,j,:]) / sqrt(head_dim)          for j <= i
    S[i, j] = -inf                                                for j >  i   (causal)
    P[i, :] = softmax_j(S[i, :])
    o[b,h,i,:] = sum_j P[i, j] * v[b,h,j,:]

i.e. `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)`
with the default scale. The causal mask is top-left aligned (query i attends
keys 0..i inclusive; q and k share seq_len). No dropout, no additive mask
tensor, no grouped-query sharing (one key/value head per query head), no
log-sum-exp side output, no backward. `causal` is True in every benchmark
case; the flag must be honoured, with `causal=False` meaning full attention.

## Inputs and outputs
- `q`, `k`, `v`: (batch, heads, seq_len, head_dim), fp16, contiguous (may be
  assumed). seq_len is a multiple of 1024 in the benchmark cases; head_dim
  is 128 and is held whole along the feature axis (no loop over head_dim is
  needed, but one is permitted).
- `causal`: Python bool (default True).
- Output: exactly one tensor with q's shape and dtype, freshly allocated
  inside run() on every call; no aliasing with any input.
- No input may be modified.
- Call form: `run(q, k, v)`, positional; `causal` keeps its default `True` and
  no keyword arguments are passed. Never run a configuration search.

## Required logical stages
1. Query block setup: the program that owns a block of query rows of one
   (batch, head) loads its query tile and initialises the running row
   statistics (row-max m, either -inf or taken from the first visited key
   tile; row-sum l = 0) and an fp32 output accumulator O = 0.
2. Streaming key/value loop over exactly the key tiles that contain at
   least one unmasked key for this query block, each visited once (tiles
   lying entirely above the diagonal are not visited when causal, because
   visiting them would roughly double the executed work). The visiting
   order is free; an order in which a row can meet a fully masked tile
   before any unmasked key needs a guard against exp(-inf - (-inf)).
   Per key tile:
   a. scores S = Q . K^T accumulated in fp32, with the scale 1/sqrt(head_dim)
      applied either to the query operand beforehand or to the scores;
   b. masked positions (key > query when causal; keys beyond seq_len if the
      tile overhangs the sequence) set to -inf;
   c. online softmax update: m_new = max(m, rowmax(S)); alpha = exp(m - m_new);
      P = exp(S - m_new); l = l * alpha + rowsum(P); O = O * alpha + P . V
      with P rounded to the input dtype before the product and fp32
      accumulation; m = m_new.
   Each tile subtracts the running maximum known at that point, with the
   rescaling above; only the final normalised result must equal the exact
   stable softmax. If the whole key range of a query block is held on chip
   at once, the loop may degenerate to a single tile: max and sum are
   evaluated over the held values, with no rescaling and no -inf initial
   state.
3. Finalisation: O = O / l, round to the output dtype, store the query block.

The running statistics and the accumulator stay on chip from stage 1 to
stage 3 and the output is the only global-memory write: a separate
score/softmax pass, a materialised score matrix, or a split-key reduction
with a cross-program combine is not permitted, because each would
round-trip scores, statistics or partial outputs through global memory
that the canonical algorithm keeps on chip. Stage 2 is a single logical
pass over K/V per query block; this counts passes of the algorithm, not
physical DRAM transactions, which caches, TMA and the compiler may change.
The number of kernel launches is not otherwise fixed.

## Algorithm family and structure
Online-softmax (flash-style) fused attention forward: tile matmul for QK^T,
running max with exponential rescaling of the running sum and accumulator,
tile matmul for PV, one final normalisation. Reductions: per query row,
a combine over the visited key tiles inside the program that owns the
query block, each step using an intra-tile row max and row sum (tree order
free; key-tile order free); the inner-product accumulation order inside
each tile matmul is free. No sort, no scan, no cross-program reduction (a
split-key combine would add a global round trip of partial outputs and
statistics). Causal tiles strictly above the diagonal are skipped (they
would add about half the work); the diagonal tile is masked with -inf.

## Precision and accumulation
- fp16 operands into both tile matmuls; fp32 accumulation of scores, of the
  running max and sum, and of the output accumulator.
- P is rounded to the input dtype (fp16) before the PV product.
- The exponent may be natural-base or base-2 with log2(e) folded into the
  scale; the scale may be folded into the query operand (with the resulting
  extra rounding to fp16) or applied to the fp32 scores.
- The final division by l may be exact or an approximate-reciprocal
  division; flush-to-zero on exponentials and the division is acceptable.
- Output rounded once to the output dtype at the store.
- Tolerance: the operator config's verify section.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Besides the launch, run() may read shapes, compute host scalars (scale,
flags), allocate the output and build optional descriptor/metadata objects
for the launch. No casts, copies, transposes, padding, re-layout or packing
of q/k/v; a `.contiguous()` call is permitted only as a no-op; nothing
cached across calls or precomputed outside run().

## Permitted implementation mappings
- Query-block and key-tile extents, launch ordering of (batch, head) versus
  query blocks, number of resident programs, pipelining depth and load
  hints.
- Loading K transposed versus transposing in registers; descriptor-based
  versus pointer-based loads.
- Applying the causal mask to every visited tile or only to tiles that
  intersect the diagonal; compile-time versus runtime loop trip count.
- Where the scale is applied and the base of the exponential (see
  Precision).
- The order in which key tiles are visited, and holding the whole key
  range of a query block on chip as a single tile (see stage 2).
- How the (batch, head, query block) work is spread over programs and
  launches, provided nothing but the output is written to global memory.

## Forbidden substitutions
- torch.nn.functional.scaled_dot_product_attention or any library attention;
  host-side torch.matmul/bmm/einsum/softmax for any stage.
- Materialising the (seq_len x seq_len) score or probability matrix, or any
  global scratch; a two-pass softmax that traverses K twice (max first, then
  exponentials); split-key partial results combined across programs (in a
  second launch or otherwise). Each adds a global round trip or a second
  logical pass over K/V that the canonical algorithm does not have.
- Processing fully masked key tiles when causal (they must be skipped, for
  example by the loop bound); omitting the -inf mask on the diagonal tile.
- fp32 operands into the matmuls or fp16 accumulators; omitting the running
  rescale when the key range is processed in more than one tile.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty_like` (or `torch.empty`) for the output.
- Tensor metadata: `.shape`, `.stride()`, `.dtype`, `.device`, `.numel()`.
- `Tensor.contiguous()` only as a no-op.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
