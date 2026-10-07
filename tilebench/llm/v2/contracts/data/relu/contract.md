# relu: canonical algorithm contract

## Functional semantics

Given a 1-D tensor `x` of length `n`, produce a new tensor `y` of the same
shape and dtype with `y[i] == max(x[i], 0)` for every element. This is the
reference `torch.relu(x)`, defined for the floating dtypes and for int8.

The entry point is called as `run(x)`; `n` is `x.numel()`. No keyword
arguments are passed.

## Inputs and outputs

- `x`: `(n,)`, contiguous, dtype one of fp16, bf16, fp32, int8. Read-only:
  `x` must not be written and the result must not alias it. The benchmark
  never supplies NaN; NaN handling and the sign of a zero result are outside
  the verified domain (results are compared by value).
- `y` (returned): `(n,)`, same dtype as `x`, freshly allocated inside `run()`
  on every call. Returned as a single tensor.
- Edge handling is required wherever a case's `n` is not a multiple
  of the chosen tile; no element outside `[0, n)` may be read or written.

## Required logical stages

1. **Allocate** the output (uninitialised allocation is sufficient).
2. **Elementwise map**: one logical traversal of `x` that reads each
   element once, replaces negative values by zero, and writes the result
   once to the same position of `y`.

Stage 2 is a single logical stage with no inter-element dependencies, and no
intermediate tensor is materialised in global memory. No additional pass over
the data is permitted, because a second pass adds a full extra read or write
of the data and so changes the algorithm rather than its mapping. These read
and write counts are logical traversals in the algorithm, not a guarantee
about physical DRAM transactions, which caches and the compiler may change.
How the index range is split across programs or launches is a mapping
choice.

## Algorithm family and structure

Flat streaming elementwise map. No reduction, scan or sort is involved. The
clamp may be expressed as a compare-and-select against zero
(`x >= 0 ? x : 0`) or as a maximum with zero; both are acceptable because
they agree for every finite input.

## Precision and accumulation

A single comparison and select (or maximum) per element in the element's own
dtype; no rounding is introduced and no dtype conversion may occur. The
zero used for the select must be of the input dtype (or be converted to it
before the store). Masked or padded tail lanes may hold any value because
they are never stored.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

There is no cast, copy, packing or cached state. The input must be consumed
as-is; no host-side copy or conversion may precede the kernel, and nothing
may be cached across calls.

## Permitted implementation mappings

- Tile length, elements per program, number of programs, vector width,
  pipelining depth and launch geometry are free.
- The number of launches is free: the index range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and no intermediate is written to global memory.
- Tail handling by explicit masks or by the DSL's bounds-padded loads and
  bounds-clipped stores is free, provided no out-of-range element is
  written. Edge handling is required wherever a case's `n` is not a
  multiple of the chosen tile; supporting shapes outside the task's configured cases is not required, and specialising per case shape (for example the shape as a compile-time constant, one specialisation per configured case) is permitted.
- The arithmetic form (select on `x >= 0`, select on `x > 0`, maximum with
  zero, or an equivalent clamp primitive inside the kernel) is free.

## Forbidden substitutions

- Computing the result with PyTorch (`torch.relu`,
  `torch.nn.functional.relu`, `torch.clamp`, `torch.clamp_min`,
  `torch.maximum`, `torch.where`, `x.clamp(min=0)`, `x * (x > 0)`,
  `y.copy_(...)`, and similar) instead of a kernel.
- Returning `x`, a view of `x`, or an in-place modification of `x`
  (`torch.relu_`, `x.clamp_`).
- Any dtype conversion of the stored result.
- Reading or writing the data in more than one logical traversal (a second
  pass over `x` or `y`), because that changes the algorithm rather than its
  mapping; this counts passes in the algorithm, not physical memory
  transactions.
- Mutating `x`.

## Permitted PyTorch operations

- `torch.empty_like(x)` for the output.
- Reading `x.numel()`, `x.shape`, `x.dtype`, `x.device`, and obtaining the
  current stream.

Everything else in `torch` is forbidden inside `run()`.
