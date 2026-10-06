# flash_decode: canonical algorithm contract

## Functional semantics
Flash-decoding STAGE 2 only: the combine of split-KV partial attention
results. Stage 1 (attention over key/value blocks) is not part of this
operator; its outputs are the inputs here. Given partial outputs `mid_o` of
shape (batch, heads, num_blocks, head_dim), per-split log-sum-exp values
`mid_o_lse` of shape (batch, heads, num_blocks), per-batch sequence lengths
`b_seqlen` of shape (batch,) and the split length `block_seq` (a 0-d CPU
int32 tensor or Python int), for every (b, h):

    nb      = ceil(b_seqlen[b] / block_seq)            (number of valid splits)
    m       = max_{s < nb} mid_o_lse[b, h, s]
    w_s     = exp(mid_o_lse[b, h, s] - m)              for s < nb, else 0
    out[b, h, :] = sum_{s < nb} w_s * mid_o[b, h, s, :] / sum_{s < nb} w_s

Splits with index >= nb are excluded. The expected value is defined with a
1e-10 added to the denominator; because the denominator is at least 1
whenever at least one split is valid, omitting the epsilon is within
tolerance and either form is accepted. Every benchmark row has
b_seqlen[b] = seq_len >= 1; rows with no valid split are unspecified.

## Inputs and outputs
- `mid_o`: (batch, heads, num_blocks, head_dim), fp32, contiguous.
- `mid_o_lse`: (batch, heads, num_blocks), fp32, contiguous.
- `b_seqlen`: (batch,), int32, on the device; the valid-split count must be
  derived from it on the device (per batch row), not assumed from shapes.
- `block_seq`: 0-d CPU int32 tensor or Python int; reading it with `.item()`
  on the host is permitted (no device synchronisation).
- Output: exactly one tensor of shape (batch, heads, head_dim) in
  `mid_o.dtype`, freshly allocated inside run() on every call; no aliasing.
- No input may be modified.
- Call form: `run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)`, positional;
  no keyword arguments are passed. Never run a configuration search.

## Required logical stages
1. For each (batch, head): read b_seqlen[b], compute nb.
2. Weighted combine over the valid splits in fp32, producing the head_dim
   numerator vector and the scalar denominator, either as a single online
   pass over the splits (running max m, rescale factor exp(m_old - m_new)
   applied to the running numerator and denominator, weight
   exp(lse_s - m_new)), or as a max pass followed by a weighted-sum pass
   over the same splits. The splits may be visited in any order and in
   chunks: each chunk subtracts the running maximum known at that point,
   with the rescaling above, and only the final normalised result must
   equal the exact stable combine. If the whole split axis is held on
   chip, the max and the weighted sums are evaluated over the held values,
   with no streaming loop and no -inf initial state. Invalid splits are
   excluded either by the loop bound or by neutralising them (weight 0,
   value 0).
3. Normalise numerator / denominator, round to the output dtype, store the
   head_dim vector.

The weights and running statistics never round-trip through global memory
and the split axis of one (batch, head) is not reduced across programs,
because the canonical combine keeps them on chip and a cross-program
combine would add a global intermediate. The number of kernel launches is
not otherwise fixed.

## Algorithm family and structure
Numerically stable log-sum-exp weighted average over the split axis (the
softmax of the per-split log-sum-exp values applied to the partial outputs).
The split axis of each (batch, head) is reduced inside one program (in any
order, streamed or held whole); the head_dim axis is element-wise and may
be held whole or tiled. No sort, no scan.

## Precision and accumulation
- fp32 for the running max, the weights, the denominator and the numerator
  accumulator; natural-base exp or an equivalent base-2 formulation.
- A single final division (with or without the 1e-10 epsilon).
- Output rounded once to the output dtype at the store.
- Tolerance: the framework's per-dtype defaults (no verify override).

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

The host `.item()` on the CPU `block_seq` scalar is permitted (it causes no
device synchronisation), as are shape reads, the output allocation and
metadata-only views of it. No casts, copies, masking or cloning of
`mid_o_lse` on the host, no padding or re-layout of `mid_o`, no state across
calls, nothing precomputed outside run().

## Permitted implementation mappings
- One (batch, head) per program (or several per program, or head_dim split
  across programs with the same per-program recurrence).
- Runtime (data-dependent) versus compile-time loop trip count over splits;
  skipping versus neutralising invalid splits.
- Online single pass versus two-pass max-then-sum over the split axis; the
  order in which splits are visited; chunked processing with rescaling
  between chunks, or holding the whole split axis on chip.
- How (batch, head) work is grouped into programs and launches.
- Vector width, pipelining, number of resident programs, specialisation on
  block_seq / num_blocks / head_dim.

## Forbidden substitutions
- Host-side tensor arithmetic for any stage (torch.exp, torch.max,
  torch.sum, torch.logsumexp, torch.softmax, masked_fill on tensors).
- Materialising the weights or masked log-sum-exp values in global memory,
  or any other global scratch (for example partial combines of a split
  range merged by a second launch).
- Recomputing attention from keys/values (there are none); ignoring
  b_seqlen (assuming all splits valid from the shape alone).
- Reduced-precision accumulation.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the output; `Tensor.view` on the output for a
  metadata-only reshape.
- `.item()` on the CPU `block_seq` scalar; `isinstance` checks on it.
- Tensor metadata: `.shape`, `.stride()`, `.dtype`, `.device`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
