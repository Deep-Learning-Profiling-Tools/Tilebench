# dropout: canonical algorithm contract

## Functional semantics
Inverted dropout applied with a PRECOMPUTED keep mask. Given `x` of shape
(n,), a mask `x_keep` of the same shape and dtype whose values are exactly
0 or 1, and a Python float `p` (drop probability):

    out[i] = x[i] * x_keep[i] * (1 / (1 - p))

i.e. `(x * x_keep) * (1.0 / (1 - p))`. Kept elements (mask 1) are scaled by
1/(1-p); dropped elements (mask 0) are exactly 0. There is NO random number
generation anywhere in this operator: the mask is an input, and run() is
fully deterministic. `p` is only used to form the rescale factor.

## Inputs and outputs
- `x`: (n,), fp16, bf16 or fp32, contiguous.
- `x_keep`: (n,), same dtype as `x`, contiguous, values in {0, 1}. It may be
  treated as a multiplicative factor or as a boolean (nonzero = keep);
  behaviour for other mask values is unspecified.
- `p`: Python float with 0 <= p < 1.
- Output: exactly one tensor with the shape and dtype of `x`, freshly
  allocated inside run() on every call; no aliasing with any input.
- No input may be modified.
- Call form: `run(x, x_keep, p)`, positional; no keyword arguments are
  passed. Never run a configuration search.

## Required logical stages
1. Element-wise masked rescale: for each element, read x and x_keep, apply
   the mask (multiply or select) and the factor 1/(1-p), round to x's dtype
   and store.

One logical pass; no reduction, no intermediate, no second pass, because the
operator is a single streaming masked rescale and an intermediate or second
pass would change its data movement. How the index range is split across
programs or launches is a mapping choice.

## Algorithm family and structure
Memory-bound streaming map reading two arrays and writing one. No reduction,
scan, sort or RNG stream. Elements are independent, so any partitioning of
the index range across programs is acceptable.

## Precision and accumulation
- Load reduced-precision inputs and compute in fp32: the mask application and
  the rescale are evaluated in fp32.
- The rescale factor may be applied as a division by (1 - p) or as a
  multiplication by the reciprocal 1/(1-p), with the reciprocal computed on
  the host (as a scalar argument) or in the kernel; the two forms differ by
  at most one fp32 ulp and are both accepted.
- Exactly one rounding, to x's dtype, at the store; dropped positions store
  exactly 0.
- Tolerance: the framework's per-dtype defaults (no verify override).

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Besides the launch, run() may perform the output allocation and host scalar
arithmetic on p. No casts, copies, transposes, packing or mask generation on
the host; no state across calls; nothing precomputed outside run(). The mask is
never generated, transformed, packed into bits or cached by run().

## Permitted implementation mappings
- Elements per program, launch geometry, vector width, number of resident
  programs.
- The number of launches is free: the index range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and nothing but the output is written
  to global memory.
- Mask as a multiplicative factor versus a boolean select.
- Division by (1 - p) versus multiplication by the reciprocal.
- Masked tail handling versus padded loads with clipped stores (padded lanes
  must never be written). Edge handling is required wherever the task's fixed
  n is not a multiple of the chosen tile; supporting shapes other than the
  task's declared shape is not required.

## Forbidden substitutions
- Any in-kernel or host-side random number generation (Philox/seed/offset
  schemes, torch.rand*, torch.bernoulli, generator seeding) or any use of a
  dropout library routine (torch.nn.functional.dropout, torch.dropout).
  The output must depend only on the supplied mask.
- Computing the result with host tensor arithmetic (x * x_keep, torch.mul,
  torch.where on tensors).
- Computing in fp16/bf16 with intermediate rounding, or changing the output
  dtype.
- Any buffer other than the output (no mask repacking, no scratch), because
  the operator consumes the supplied mask as given and writes only the
  output.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty_like` (or `torch.empty`) for the output.
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
