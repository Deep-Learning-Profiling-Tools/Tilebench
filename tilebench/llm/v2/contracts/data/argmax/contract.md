# argmax: canonical algorithm contract

## Functional semantics

Row-wise argmax of a 2-D matrix, exactly `torch.argmax(x, dim=1)`:

```
out[m] = the smallest n such that x[m, n] == max over n' of x[m, n']
```

Ties resolve to the first (lowest-index) occurrence. The result is an int64
vector of length `M`. NaN handling is unspecified; the benchmark inputs
contain no NaN.

The entry point is called as `run(x, dim, **kwargs)`. `dim` is `1` in every
benchmark case; the implementation may assert `dim == 1` or handle other
values by any means, but the `dim == 1` path must not copy `x`.
The entry point is called positionally with exactly the inputs listed below;
no keyword arguments are passed (declared keyword parameters, if any, must
have defaults and must not affect the computation).

## Inputs and outputs

- `x`: `(M, N)`, contiguous row-major, dtype fp16 or fp32, on the GPU.
  Read-only.
- Output: `(M,)`, dtype int64, freshly allocated inside `run()` on every
  call, returned as a single tensor. It must not alias `x`.

## Required logical stages

1. **Allocate** the `(M,)` int64 output (uninitialised allocation suffices).
2. **Row reduction**: one device pass in which every row of `x` is read
   exactly once and reduced to the index of its first maximum, which is
   written once.

Stage 2 is the only device work and is a single launch. The whole of a row
must be reduced by the program that owns it; no partial (value, index)
results may be written to scratch and merged by a further pass. A two-pass
scheme (find the row maximum, then search for its first index) reads `x`
twice and is forbidden.

## Algorithm family and structure

Sequential chunked scan per row. A program walks its row in column chunks in
increasing column order, keeping a running best value and best index. For
each chunk it computes the chunk maximum and the lowest index attaining it
(a tree or sequential reduction over the chunk), then replaces the running
best only when the chunk maximum is strictly greater than the running best
value, so an earlier chunk keeps the win on ties. Combined with the
lowest-index rule inside a chunk this yields the first occurrence over the
row. Columns beyond `N` in the last chunk must be padded with `-inf` so they
never win; the running best value starts at `-inf` and the running index at
`0`, so an all-`-inf` row yields index `0`, as the reference does. Index
arithmetic (`chunk_start + local_index`) must be exact in int64 (or in a
type that cannot overflow for the given `N`).

## Precision and accumulation

No arithmetic beyond comparisons. Values may be compared in the input dtype
or after an exact upcast to fp32 (both preserve ordering); the running best
value must have at least the input's range so that comparisons are exact.
Indices are produced as int64. `-inf` is the only fill value.

## Preprocessing and timing boundary

Everything `run()` does is timed. For the benchmark's contiguous `x` with
`dim == 1`, no copy, cast or transpose may be made and nothing may be cached
across calls. It may check device/contiguity, allocate the output, compute
the chunk count, and launch. The metric formulas count one read of `x`; the
`8*M`-byte output write is not counted and is negligible.

## Permitted implementation mappings

- Chunk width, number of chunks, programs per launch, vector width,
  pipelining, compile-time versus runtime chunk counts.
- Whether the in-chunk max/argmax is a tree reduction primitive or an
  explicit scan, provided it returns the lowest index on ties.
- Whether values are upcast to fp32 before comparison.
- Masked loads with `-inf` fill, or bounds-padded block loads with a
  `-inf` padding mode, for the tail chunk.

## Forbidden substitutions

- `torch.argmax`, `torch.max` with `dim`, `torch.topk`, `torch.sort` /
  `torch.argsort`, or any other library reduction producing the result.
- Any scheme that reads `x` more than once, or that materialises per-chunk
  partial results in global scratch.
- A merge rule that lets a later chunk win on ties (for example `>=`), or
  an in-chunk reduction that returns the last maximal index.
- A fill value other than `-inf` for out-of-range columns; mutating `x`;
  returning a view of `x`.

## Permitted PyTorch operations

- `torch.empty(M, dtype=torch.int64, device=x.device)` for the output.
- `x.contiguous()` only as a no-op guard on the already-contiguous input.
- Reading dtype, shape, stride and device metadata (`x.is_cuda`,
  `x.shape`), and obtaining the current stream.

Everything else in `torch` is forbidden inside `run()`.
