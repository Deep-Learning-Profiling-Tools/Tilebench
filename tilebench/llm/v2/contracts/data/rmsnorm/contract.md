# rmsnorm: canonical algorithm contract

## Functional semantics

Root-mean-square normalisation over the last dimension with a learned
per-feature weight. For an input `x` of shape `(batch, M, K)` and a weight
`w` of shape `(K,)`, every row `r = (b, m)` is normalised independently:

    ms[r]    = (1 / K) * sum_k x[r, k]^2
    rstd[r]  = 1 / sqrt(ms[r] + eps)
    y[r, k]  = x[r, k] * rstd[r] * w[k]

This is the reference `torch.nn.functional.rms_norm(x, x.shape[-1:],
weight=w, eps=eps)`. `eps` defaults to `1e-6` and the benchmark never
overrides it. The mean divides by the true row length `K`, never by a padded
length. There is no reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(x, rms_w)`: `x` and `rms_w` positional;
`eps` keeps its default of `1e-6` and no keyword arguments are passed.

- `x`: `(batch, M, K)`, contiguous row-major, dtype one of fp16, bf16, fp32.
  Read-only.
- `rms_w`: `(K,)`, contiguous, same dtype as `x`. Read-only.
- `eps`: Python float, added inside the square root.
- Returned: one new tensor of shape `(batch, M, K)` and dtype of `x`,
  allocated inside `run()` on every call. It must not alias or be a view of
  either input.

## Required logical stages

1. **Row statistic**: for each row, the sum of squares over `K` in fp32,
   divided by `K`, plus `eps`, then the reciprocal square root: one scalar
   `rstd` per row. Depends on the whole row.
2. **Normalise and scale**: `y = x * rstd * w` for every element of the row,
   evaluated in fp32 and cast once to the output dtype at the store.

Stage 2 depends on stage 1 for the same row only. The two stages are
normally fused, with `rstd` held privately by the program that owns the row
(re-reading the row for stage 2 or keeping it on chip are both permitted).
Splitting them into two launches that pass `rstd` through a `(batch*M,)`
fp32 statistics buffer allocated inside `run()` on every call is a mapping
choice: the buffer's traffic is device work and is counted, and it must
never be cached across calls. What is fixed is the logical traversal count:
the row is read once for the statistic and at most once more for the
normalisation; a split that re-reads the whole row from global memory a
third time is not permitted. Splitting one row across several programs
(cross-program reduction, atomics) is not permitted, because combining the
partial sums would need a global-memory intermediate or atomics that the
canonical algorithm does not have. Rows are independent; several rows per
program, a chunked loop over a long row, or distributing the rows over more
than one launch is free.

## Algorithm family and structure

Two-pass row normalisation: a reduction sweep (sum of squares) followed by
an elementwise sweep (normalise, scale), each row handled whole by one
program in each sweep. The reduction
is a plain fp32 sum over `K` elements; the grouping of partial sums
(chunk-sequential, lane-wise accumulation, tree over lanes, or any other
order) is free. There is no scan and no sort. Out-of-range lanes of a partial
chunk contribute exactly zero to the sum of squares and are never stored. The
second sweep may re-read the row from memory or keep it on chip when it fits.

## Precision and accumulation

- Every loaded element of `x` and `w` is upcast to fp32 before use, for all
  three input dtypes.
- The sum of squares, the division by `K`, the `eps` addition, the
  reciprocal square root and the product `x * rstd * w` are all fp32.
- A single downcast to the input dtype happens at the final store; no
  intermediate rounding to fp16/bf16.
- Tolerances are the verifier's per-dtype defaults (no `verify` override in
  the config); approximate versus exact reciprocal-square-root lowering is
  accepted within them.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Viewing `x` as `(batch*M, K)` is metadata only. No input repacking, no cast
copies, no state cached across calls, nothing precomputed outside `run()`.
In a split design the per-call statistics buffer is written and read by
device work and is counted. The weight is consumed as given, a contiguous
`(K,)` vector, and may be re-read per row.

## Permitted implementation mappings

- Chunk width along `K`, rows per program, vectorisation, pipelining depth
  and other launch parameters.
- Fused versus split launches: distributing the rows over one or several
  launches, or computing `rstd` in one launch and applying it in another
  through a per-call `(batch*M,)` fp32 statistics buffer.
- Re-reading the row for the second sweep versus holding it on chip.
- Internal order of the fp32 sum.
- Fill values for out-of-range lanes of `x` (must act as zero in the sum) and
  of `w` (any value, since padded lanes are never stored).
- Whether `K` is a compile-time constant or a runtime argument; whether
  explicit strides are passed or the contiguous layout is assumed.

## Forbidden substitutions

- Calling `torch.nn.functional.rms_norm`, `torch.layer_norm`,
  `F.layer_norm` or any library normalisation routine.
- Computing the row statistic with PyTorch tensor operations on the host
  (`x.pow(2).mean(-1)`, `torch.rsqrt`, `torch.mean`, `torch.sum`, ...) and
  only applying the scale in a kernel.
- Accumulating the sum of squares, or evaluating the normalise multiply, in
  fp16/bf16.
- Dividing by a padded row length instead of `K`, or omitting `eps`.
- Splitting one row's reduction across programs (combining the partial sums
  would add a global-memory intermediate or atomics that the canonical
  algorithm does not have); a split that re-reads the whole row from global
  memory a third time; caching a statistics buffer across calls.
- Writing the result into `x` in place, or returning a view of `x`.

## Permitted PyTorch operations

- `torch.empty_like(x)` or `torch.empty(...)` for the output and, only in a
  split design, one fp32 `(batch*M,)` statistics buffer allocated per call.
- `Tensor.reshape` / `Tensor.view` to the 2-D `(batch*M, K)` form,
  `Tensor.contiguous` as a no-op guard on the contiguous inputs,
  `Tensor.stride`, `Tensor.numel`, `.shape`, `.dtype`, `.device`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`: no arithmetic,
reduction, indexing or conversion of tensors through PyTorch.
