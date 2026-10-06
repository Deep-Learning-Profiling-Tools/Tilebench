# quantize_global: canonical algorithm contract

## Functional semantics

Given a 1-D float32 tensor `x` of length `n`, produce a new float16 tensor
`out` of the same shape with `out[i] == float16(x[i])`, where the conversion
is the standard IEEE narrowing with round-to-nearest-even. This is the
reference `x.to(torch.float16)`.

The operator name is historical: no scale factor, no global absolute
maximum, no clamping and no integer quantisation is part of this operator;
the whole computation is one elementwise dtype conversion. An implementation
must not add any of those steps.

The entry point is called as `run(x)`; `n` is `x.numel()`. No keyword
arguments are passed.

## Inputs and outputs

- `x`: `(n,)`, contiguous, dtype float32 (the benchmark uses only float32
  input). Read-only: `x` must not be written and the result must not alias
  it. The benchmark's values lie well inside the float16 range, so overflow
  to infinity is outside the verified domain.
- `out` (returned): `(n,)`, dtype float16, freshly allocated inside `run()`
  on every call. Returned as a single tensor.
- Edge handling is required wherever the task's fixed `n` is not a multiple
  of the chosen tile length; no element outside `[0, n)` may be read or
  written. Supporting lengths other than the task's declared `n` is not
  required.

## Required logical stages

1. **Allocate** the float16 output (uninitialised allocation is sufficient).
2. **Elementwise convert**: in one logical traversal, read each float32
   element, convert it to float16, and write it to the same position of
   `out`.

Stage 2 is a single logical stage with no inter-element dependencies; how it
is distributed over launches is a mapping choice. There is no reduction
stage of any kind; adding a preliminary pass over the data (for example to
compute a maximum) is forbidden, because no maximum or scale is part of this
operator. Pass counts are logical traversals in the algorithm, not
guarantees about physical DRAM transactions, which caches, TMA and the
compiler may change.

## Algorithm family and structure

Flat streaming elementwise dtype conversion. No reduction, scan or sort is
involved.

## Precision and accumulation

- Input float32, output float16, one conversion per element with
  round-to-nearest-even; the result must match the reference conversion
  within the verification tolerance.
- No scaling, offset, clamping or saturation may be applied before or after
  the conversion.
- Masked or padded tail lanes may hold any value because they are never
  stored.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

There is no cast, copy or packing of `x` on the host, no scale computation,
and nothing cached across calls. `x.contiguous()` may be called only as a
no-op guard on the already-contiguous input.

## Permitted implementation mappings

- Tile length, elements per program, number of programs, number of launches,
  vector width, pipelining depth and launch geometry are free.
- Tail handling by explicit masks or by the DSL's bounds-padded loads and
  bounds-clipped stores is free, provided no out-of-range element is
  written.
- The DSL's native float32-to-float16 conversion primitive, or any
  equivalent that yields the round-to-nearest-even result, may be used.

## Forbidden substitutions

- Performing the conversion with PyTorch (`x.to(torch.float16)`,
  `x.half()`, `out.copy_(x)`, `torch.Tensor.type`, and similar) instead of a
  kernel.
- Adding an absolute-maximum reduction, a scale factor, a second output, an
  int8 or other integer output, or any clamping.
- More than one logical traversal of the input or of the output (a count of
  passes in the algorithm, not of physical memory transactions).
- Returning `x`, a view of `x`, or a float32 result.
- Mutating `x`.

## Permitted PyTorch operations

- `torch.empty(x.shape, dtype=torch.float16, device=x.device)` for the
  output.
- `x.contiguous()` only as a no-op guard on the already-contiguous input.
- Reading `x.numel()`, `x.shape`, `x.dtype`, `x.device`, and obtaining the
  current stream.

Everything else in `torch` is forbidden inside `run()`.
