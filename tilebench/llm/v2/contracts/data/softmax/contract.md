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
   denominator `l = sum exp(x - m)`, both in fp32, obtained in a single
   streaming sweep over the row with the online rescaling recurrence
   (running max `m`, running sum `l`; on a new chunk with maximum `m_c`:
   `m_new = max(m, m_c)`, `l = l * exp(m - m_new) + sum exp(chunk - m_new)`,
   `m = m_new`). Depends on the whole row.
2. **Normalise**: for every element, `y = exp(x - m) / l` in fp32, cast once
   to the output dtype at the store.

Stage 2 depends on stage 1 for the same row only. Both stages belong to one
launch: the statistics are private values of the program that owns the row
and never touch global memory. Splitting one row across several programs
(cross-program max/sum, atomics) or splitting the two stages into separate
launches with global `m`/`l` buffers is not permitted. Several rows per
program, or a chunked loop over a long row, is free. When a whole row is
held on chip, the order in which `m` and `l` are evaluated over the held
values is free, but the row may be read from memory at most twice (once for
the statistics, once for the normalisation).

## Algorithm family and structure

Online (streaming, rescaling) two-sweep row softmax inside one program:
sweep one yields both statistics, sweep two normalises. Within a chunk the
max and the sum are tile-wide reductions whose internal order is free;
across chunks the recurrence is sequential. The initial state is
`m = -inf, l = 0`. Out-of-range lanes of a partial chunk must behave as
`-inf` in sweep one (they contribute `exp(-inf) = 0` to the sum and never
win the max) and must never be stored in sweep two. The classic three-sweep
scheme (max sweep, then sum sweep, then normalise) is not the canonical
structure.

## Precision and accumulation

- Every loaded element is upcast to fp32 before use, for both input dtypes.
- `m`, `l`, every exponential and the final division are fp32.
- A single downcast to the input dtype happens at the final store.
- The natural exponential may be lowered as `exp` or as `exp2` with a
  `log2(e)` scale; a reciprocal-multiply in place of the division is
  accepted. Tolerances are the verifier's per-dtype defaults (no `verify`
  override).

## Preprocessing and timing boundary

Everything `run()` does is timed: the output allocation and the launch. `x`
is consumed as given; no copy, cast, transpose, padding, cached state or
precomputation outside `run()`.

## Permitted implementation mappings

- Chunk width along the row, rows per program, vectorisation, pipelining
  depth and other launch parameters.
- Re-reading the row for sweep two versus holding it on chip; a single
  sweep when the whole row fits in one chunk.
- `exp` versus `exp2` formulation; division versus reciprocal multiply;
  internal order of the per-chunk max and sum.
- Whether the number of chunks or the row length is a compile-time constant
  or a runtime argument; explicit strides versus the contiguous layout.

## Forbidden substitutions

- Calling `torch.softmax`, `torch.nn.functional.softmax`, `x.softmax(...)`,
  `log_softmax`, `nn.Softmax`, or any library softmax routine.
- Computing `m` or `l` with PyTorch reductions on the host (`torch.max`,
  `torch.amax`, `torch.sum`, `torch.exp`, `torch.logsumexp`, ...).
- Omitting the max subtraction, or subtracting anything other than the
  exact row maximum in the final normalisation.
- Evaluating the exponentials or the sum in fp16.
- A third sweep over the row from memory; splitting a row across programs
  or launches; global scratch for the statistics.
- Writing into `x` in place or returning a view of `x`.

## Permitted PyTorch operations

- `torch.empty_like(x)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.stride`, `Tensor.numel`;
  `Tensor.contiguous` as a no-op guard on the contiguous input.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
