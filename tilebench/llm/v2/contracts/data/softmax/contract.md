# softmax: canonical algorithm contract

## Functional semantics

Row-wise, numerically stable softmax over the last dimension of a 2-D
tensor `x` of shape `(n_rows, n_cols)`:

    m[i]    = max_j x[i, j]
    l[i]    = sum_j exp(x[i, j] - m[i])
    y[i, j] = exp(x[i, j] - m[i]) / l[i]

This is the reference `torch.softmax(x, dim=-1)`. Rows are independent.
There is no reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(x)`: `x` positional; no keyword arguments
are passed.

- `x`: exactly 2-D, `(n_rows, n_cols)`, contiguous row-major, dtype fp16 or
  fp32. Read-only.
- Returned: one new tensor of shape `(n_rows, n_cols)` and dtype of `x`,
  allocated inside `run()` on every call. It must not alias `x`.

## Required logical stages
1. **Row statistics**: for each row, the exact row maximum `m` and the
   denominator `l = sum_j exp(x[i, j] - m)`, both in fp32. When the row is
   processed in chunks, they are obtained in one logical traversal of the
   row with the online rescaling recurrence (running max `m`, running sum
   `l`; on a new chunk with maximum `m_c`: `m_new = max(m, m_c)`,
   `l = l * exp(m - m_new) + sum exp(chunk - m_new)`, `m = m_new`).
   Subtracting the running maximum inside a chunk is part of this
   recurrence and is correct: only the final `m` and `l` must be the exact
   row statistics. When the whole row is held on chip, `m` and `l` may be
   evaluated directly over the held values (the recurrence then degenerates
   to one chunk: no streaming loop, no repeated load and no literal `-inf`
   initial state are required). Depends on the whole row.
2. **Normalise**: for every element, `y = exp(x - m) / l` with the final
   row statistics, in fp32, cast once to the output dtype at the store.

Stage 2 depends on stage 1 for the same row only. `m` and `l` are private
values of the program that owns the row and never pass through global
memory, because a global round trip of the statistics (cross-program
max/sum, atomics, or a separate statistics launch writing `m`/`l` buffers)
adds traffic and a dependency that the canonical algorithm does not have.
Several rows per program, or a chunked loop over a long row, is free. The
algorithm traverses each row at most twice logically (once for the
statistics, once for the normalisation, or once in total when the row is
held on chip); this counts logical traversals of the input in the
algorithm, not physical DRAM transactions, which caches, TMA and the
compiler may change.

## Algorithm family and structure
Online (rescaling) two-traversal row softmax inside the program that owns
the row: the first traversal yields both statistics, the second
normalises; a row held on chip needs a single load. Within a chunk the max
and the sum are tile-wide reductions whose internal order is free; across
chunks the recurrence is sequential. When the recurrence is used over
several chunks its initial state is `m = -inf, l = 0` (or the statistics
of the first chunk). Out-of-range lanes of a partial chunk (only when a case's row length is not a multiple of the chosen chunk width) must
behave as `-inf` in the statistics (they contribute `exp(-inf) = 0` to the
sum and never win the max) and must never be stored. The classic
three-traversal scheme (a max traversal, then a sum traversal, then the
normalisation) is not the canonical structure, because it adds a third
logical traversal of every row.

## Precision and accumulation

- Every loaded element is upcast to fp32 before use, for both input dtypes.
- `m`, `l`, every exponential and the final division are fp32.
- A single downcast to the input dtype happens at the final store.
- The natural exponential may be lowered as `exp` or as `exp2` with a
  `log2(e)` scale; a reciprocal-multiply in place of the division is
  accepted. Tolerances are the verifier's per-dtype defaults (no `verify`
  override).

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

`x` is consumed as given; no copy, cast, transpose or padding of `x`, no
state cached across calls and no precomputation outside `run()`. The
output is allocated inside `run()` on every call.

## Permitted implementation mappings
- Chunk width along the row, rows per program, grid shape, vectorisation,
  pipelining depth and other launch parameters; the kernel may be
  specialised per case shape (supporting shapes outside the configured cases is not required).
- Holding the whole row on chip (single load, direct max/sum) versus a
  chunked online traversal followed by a second traversal for the
  normalisation.
- `exp` versus `exp2` formulation; division versus reciprocal multiply;
  internal order of the per-chunk max and sum.
- Whether the number of chunks or the row length is a compile-time constant
  or a runtime argument; explicit strides versus the contiguous layout.

## Forbidden substitutions
- Calling `torch.softmax`, `torch.nn.functional.softmax`, `x.softmax(...)`,
  `log_softmax`, `nn.Softmax`, or any library softmax routine.
- Computing `m` or `l` with PyTorch reductions on the host (`torch.max`,
  `torch.amax`, `torch.sum`, `torch.exp`, `torch.logsumexp`, ...).
- Omitting the max subtraction, or normalising with anything other than the
  final exact row statistics (subtracting a running maximum inside the
  recurrence is permitted).
- Evaluating the exponentials or the sum in fp16.
- A third logical traversal of the row; passing statistics or partial sums
  between programs or launches through global memory.
- Writing into `x` in place or returning a view of `x`.

## Permitted PyTorch operations

- `torch.empty_like(x)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.stride`, `Tensor.numel`;
  `Tensor.contiguous` as a no-op guard on the contiguous input.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
