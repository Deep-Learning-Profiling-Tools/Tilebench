# mean_reduction: canonical algorithm contract

## Functional semantics

Given a 2-D tensor `x` of shape `(M, N)`, compute the arithmetic mean of each
row in float32:

```
out[r] = (1 / N) * sum_{c = 0}^{N - 1} float32(x[r, c])      for r in [0, M)
```

This is the reference `x.mean(dim=1, dtype=torch.float32)`: the reduction and
the result are float32 regardless of the input dtype.

The entry point is called as `run(x, dim, **kwargs)`. The benchmark always
passes a 2-D `x` and `dim == 1` (reduce the contiguous last axis). Other
values of `dim` are outside the benchmark; an implementation may reject them
or handle them, but any re-layout it performs for them happens inside
`run()` and is timed. The entry point is called positionally; no keyword
arguments are passed.

## Inputs and outputs

- `x`: `(M, N)`, contiguous row-major (row stride `N`, element stride 1),
  dtype one of fp16, bf16, fp32. Read-only: `x` must not be written.
- `out` (returned): shape `(M,)`, dtype float32, allocated inside `run()` on
  every call. Returning a metadata-only view (for example a squeeze of a
  run-allocated `(M, 1)` buffer) is acceptable; the returned tensor must not
  share storage with `x`. Returned as a single tensor.
- Edge handling is required wherever the task's fixed `N` is not a multiple
  of the chosen tile width; columns outside `[0, N)` must
  contribute exactly zero to the sum and the divisor must be the exact `N`.

## Required logical stages

1. **Allocate** the float32 output.
2. **Row-wise sum**: for each row, traverse every element exactly once (one
   logical traversal), convert it to float32 and accumulate it into a
   float32 partial sum; combine the partial sums of the row into one
   float32 row total.
3. **Normalise and store**: divide the row total by `N` in float32 and write
   one float32 value per row.

Stages 2 and 3 are fused per row; each row is independent of every other
row. Splitting a row's reduction into a partial-sums pass followed by a
combine pass through a global scratch buffer is not required by the data
and is not permitted, because the canonical algorithm keeps the partial
sums on chip: the operator is a single logical traversal of `x` with no
global intermediates. The traversal count is a property of the algorithm,
not a guarantee about physical DRAM transactions, which caches, TMA and the
compiler may change. The number of kernel launches is not otherwise fixed.

## Algorithm family and structure

Single-logical-pass row-wise reduction. Within a row the sum may be organised
in any order: lane-wise partial sums over column chunks followed by a
cross-lane tree, a direct tree, or a sequential loop. The result must be
within the verification tolerance of the reference float32 mean, not
bit-identical to it. No reduction across rows exists. No scan or sort is
involved.

## Precision and accumulation

- Each input element must be converted to float32 before it is added; the
  accumulator and the row total are float32 for all three input dtypes.
  Accumulating in the input dtype (fp16 or bf16) is forbidden.
- The mean is `row_total / N` computed in float32. Multiplying by a
  precomputed float32 reciprocal of `N` is a rounding-level change and is
  tolerated.
- Padded or masked columns must contribute `0.0`.
- The output is stored as float32 without any narrowing.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

For the benchmark input (`x` 2-D and contiguous, `dim == 1`) no copy, cast,
transpose or padding of `x` may be made before the kernel reads it, and
nothing may be cached across calls. `x.contiguous()` may be called only as a
no-op guard.

## Permitted implementation mappings

- Rows per program (one or several), column chunk width, number of lanes,
  pipelining depth, vector width and launch geometry are free.
- Whether `N` is specialised as a compile-time constant or passed at runtime
  is free; edge handling is required wherever the task's fixed `N` is not a
  multiple of the chosen chunk width, and supporting shapes other than the
  task's declared shape is not required.
- The order in which float32 partial sums are combined is free.
- Allocating the output as `(M,)` or `(M, 1)` float32 and returning a view of
  shape `(M,)` is free.
- Tail columns may be handled by explicit masks with a zero fill or by the
  DSL's zero-padded loads.

## Forbidden substitutions

- Computing the mean or the row sums with PyTorch (`x.mean`, `torch.mean`,
  `x.sum`, `torch.sum`, `x.float().mean`, `torch.einsum`, matrix-vector
  products with a ones vector, and similar).
- Accumulating in a dtype narrower than float32.
- More than one logical traversal of `x` (a second pass over its
  elements), or staging partial sums in global memory.
- Casting or copying `x` on the host before the kernel.
- Mutating `x`.

## Permitted PyTorch operations

- `torch.empty(M, dtype=torch.float32, device=x.device)` or
  `torch.empty((M, 1), ...)` for the output.
- `.squeeze(1)`, `.view(M)` or `.reshape(M)` on the run-allocated output
  (metadata only).
- `x.contiguous()` only as a no-op guard on the already-contiguous input.
- Reading `x.shape`, `x.ndim`, `x.stride()`, `x.dtype`, `x.device`, and
  obtaining the current stream.

Everything else in `torch` is forbidden inside `run()`.
