# mul2: canonical algorithm contract

## Functional semantics

Given a 1-D tensor `x` of length `n`, produce a new tensor `y` of the same
shape and dtype with `y[i] == 2 * x[i]` for every element. This is the
reference `x * 2` with a Python scalar, which keeps the tensor dtype.

The entry point is called as `run(x)`; `n` is `x.numel()`. No keyword
arguments are passed.

## Inputs and outputs

- `x`: `(n,)`, contiguous, dtype one of fp16, bf16, fp32, int8. Read-only:
  `x` must not be written and the result must not alias it. For int8 the
  benchmark supplies values whose doubled value stays within the int8 range,
  so overflow behaviour is outside the verified domain.
- `y` (returned): `(n,)`, same dtype as `x`, freshly allocated inside `run()`
  on every call. Returned as a single tensor.
- `n` need not be a multiple of any tile length; no element outside `[0, n)`
  may be read or written.

## Required logical stages

1. **Allocate** the output (uninitialised allocation is sufficient).
2. **Elementwise map**: read each element of `x` exactly once, double it, and
   write the result exactly once to the same position of `y`.

Stage 2 is a single logical stage with no inter-element dependencies and is
expected to be one launch. No additional pass over the data is permitted.

## Algorithm family and structure

Flat streaming elementwise map. No reduction, scan or sort is involved. The
multiply may be expressed as `x * 2`, `x + x`, or for integer dtypes a left
shift by one; all are exact for the verified input domain.

## Precision and accumulation

A single operation per element in the element's own dtype (or in a wider
intermediate that is narrowed back to the input dtype on store). Doubling is
exact for fp16, bf16 and fp32 (an exponent increment, with overflow to inf
only for magnitudes the benchmark never produces) and exact for the bounded
int8 inputs. No rounding beyond this may be introduced; the output dtype
equals the input dtype. Masked or padded tail lanes may hold any value
because they are never stored.

## Preprocessing and timing boundary

Everything happens inside `run()` and is timed: the output allocation and
the kernel. There is no cast, copy, packing or cached state. The input must
be consumed as-is; no host-side copy or conversion may precede the kernel,
and nothing may be cached across calls.

## Permitted implementation mappings

- Tile length, elements per program, number of programs, vector width,
  pipelining depth and launch geometry are free.
- Tail handling by explicit masks or by the DSL's bounds-padded loads and
  bounds-clipped stores is free, provided no out-of-range element is
  written.
- The arithmetic form of the doubling (`* 2`, `+ x`, shift for integers) is
  free.

## Forbidden substitutions

- Computing the result with PyTorch (`x * 2`, `x + x`, `torch.mul`,
  `torch.add`, `x.mul(2)`, `y.copy_(x * 2)`, `torch.lerp`, and similar)
  instead of a kernel.
- Returning `x`, a view of `x`, or an in-place modification of `x`.
- Any dtype conversion of the stored result.
- Reading or writing the data more than once.
- Mutating `x`.

## Permitted PyTorch operations

- `torch.empty_like(x)` for the output.
- Reading `x.numel()`, `x.shape`, `x.dtype`, `x.device`, and obtaining the
  current stream.

Everything else in `torch` is forbidden inside `run()`.
