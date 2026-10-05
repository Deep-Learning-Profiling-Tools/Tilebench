# cross_entropy: canonical algorithm contract

## Functional semantics
Per-row softmax cross-entropy without reduction. Given `logits` of shape
(batch_size, num_classes) and integer `targets` of shape (batch_size,), for
every row i:

    m_i     = max_j logits[i, j]
    s_i     = sum_j exp(logits[i, j] - m_i)
    loss[i] = -(logits[i, targets[i]] - m_i - log(s_i))

This equals `torch.nn.functional.cross_entropy(logits, targets, reduction="none")`:
no label smoothing, no ignore_index, no class weights, no reduction across
rows. Targets are guaranteed to lie in [0, num_classes); behaviour for
out-of-range targets is unspecified and need not be handled.

## Inputs and outputs
- `logits`: (batch_size, num_classes), fp16 or fp32, row-major. It may be
  assumed contiguous, or its strides may be honoured.
- `targets`: (batch_size,), int64. Read as-is; indices may be narrowed to
  int32 for addressing.
- Output: exactly one tensor of shape (batch_size,) in `logits.dtype`,
  freshly allocated inside run() on every call. It must not alias any input.
- No input may be modified.
- Call form: `run(logits, targets)`, positional; no keyword arguments are
  passed. Never run a configuration search.

## Required logical stages
1. Row statistics: for each row, the maximum over the class axis and the sum
   over classes of exp(logit - max), from a single read of the row.
2. Target gather: the logit at column targets[i] of the same row, obtained
   either by a second one-element read or by extraction from the row data
   already loaded.
3. Loss: loss[i] = -(target_logit - max - log(sum)), rounded to the output
   dtype and stored as one scalar per row.

Stage 3 depends on stages 1 and 2; stages 1 and 2 are independent. All three
stages must execute in one launch (one row per program, or several rows per
program). Splitting them across launches with a materialised intermediate
(shifted logits, probabilities, row statistics) is not permitted.

## Algorithm family and structure
Row-wise fused log-sum-exp plus gather in the numerically stable form (the row
maximum is subtracted before exponentiation). Two reductions per row over the
class axis: a max, then a sum of shifted exponentials. The class axis may be
held in one logical tile padded to a power of two, in which case padding
lanes must hold -inf so they contribute exp(-inf) = 0 to the sum and never
win the max; or it may be processed in chunks with an online update (running
max with rescaling of the running sum). Either way the result is the stable
log-sum-exp of the complete row. The reduction tree order inside a tile is
free. No sort, no scan, no reduction across rows.

## Precision and accumulation
- All arithmetic (max, subtraction, exp, sum, log, final negation) in fp32
  regardless of the input dtype: reduced-precision logits are converted to
  fp32 after loading.
- Natural-base exp/log, or base-2 intrinsics with the appropriate log2(e)
  factors, provided the value computed is the natural log-sum-exp.
- The fp32 loss is rounded exactly once, to the output dtype, at the store.
- Tolerance: the framework's per-dtype defaults (the operator config has no
  verify override).

## Preprocessing and timing boundary
run() performs only: reading shapes/strides, allocating the output, computing
host scalars (e.g. a padded tile width), and launching. No casts, copies,
transposes, padding or packing of inputs on the host; no state cached across
calls; nothing precomputed outside run(). A `.contiguous()` call on `logits`
is permitted only as a no-op (the inputs are already contiguous).

## Permitted implementation mappings
- Number of rows per program and the launch geometry.
- Whole-row tile (power-of-two padded) versus chunked class axis with an
  online update.
- Re-reading the target logit from memory versus selecting it from the loaded
  row.
- Honouring strides versus assuming contiguity; vector width; pipelining;
  number of concurrently resident programs.

## Forbidden substitutions
- Any library cross-entropy, log-softmax, softmax, log-sum-exp or gather
  executed on the host (torch.nn.functional.*, torch.logsumexp,
  torch.log_softmax, torch.softmax, torch.gather, torch.max/torch.sum over
  tensors).
- Materialising probabilities, shifted logits or row statistics in global
  memory; any multi-launch structure with a global intermediate.
- Computing the statistics in fp16/bf16; omitting the max subtraction.
- Reducing across rows or returning a scalar/mean.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the output tensor.
- Tensor metadata: `.shape`, `.stride()`, `.dtype`, `.device`, `.numel()`.
- `torch.cuda.current_stream()` to obtain the launch stream.
- `Tensor.contiguous()` on `logits` only (expected to be a no-op).
Everything else is forbidden.
