# vector_add: canonical algorithm contract

## Functional semantics

Elementwise addition of two 1-D tensors of `n` elements with the same
dtype:

    out[i] = x[i] + y[i]        for 0 <= i < n

This is the reference `x + y`. For int8 the sum follows PyTorch's integer
semantics (wraparound on overflow); the benchmark generator keeps int8
inputs in a range where no overflow occurs, so the result must equal the
exact integer sum. There is no reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(x, y)`: `x`, `y` positional; no keyword
arguments are passed.

- `x`: `(n,)`, contiguous, dtype one of fp16, bf16, fp32, int8. Read-only.
- `y`: `(n,)`, contiguous, same dtype and length as `x` (not validated by
  the benchmark). Read-only.
- Returned: one new tensor of `x`'s shape and dtype, allocated inside
  `run()` on every call. It must not alias either input.

## Required logical stages

1. **Allocate** the output.
2. **Elementwise add**: load a contiguous block of `x` and of `y`, add,
   store under a bounds mask.

Stage 2 is a single logical stage with no inter-block dependency and is
expected to be one launch. No second pass over the data is permitted.

## Algorithm family and structure

Memory-bound 1-D elementwise binary map with flat contiguous blocking: two
reads and one write per element. No reduction, scan or sort.

## Precision and accumulation

- The add is performed in the input dtype (fp16 in fp16, bf16 in bf16,
  fp32 in fp32, int8 in int8), exactly as the reference does. Computing in a
  wider type and casting back once is accepted for the floating dtypes
  (within the verifier's per-dtype defaults) and for int8 provided the
  cast reproduces the exact int8 sum.
- The verifier compares integer dtypes exactly; floating dtypes use the
  per-dtype default tolerances (no `verify` override).
- No accumulation.

## Preprocessing and timing boundary

Everything `run()` does is timed: the output allocation and the launch.
The inputs are consumed as given; no copy, cast, cached state or
precomputation outside `run()`.

## Permitted implementation mappings

- Elements per program, grid shape, pipelining, vector width and other
  launch parameters.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for a last partial block (the benchmark sizes are multiples of common
  block widths, but correctness for arbitrary `n` is required: no element at
  index `>= n` may be read or written).
- Whether `n` is a compile-time constant or a runtime argument.

## Forbidden substitutions

- Returning `x + y`, `torch.add`, `x.add(y)`, `out.copy_(x + y)` or any
  other PyTorch arithmetic on the tensors inside `run()`.
- In-place accumulation into an input (`x.add_(y)`, `x += y`) or
  returning a view of an input.
- A dtype conversion that changes the rounding of the result (for example
  adding fp16 inputs in fp16 after an intermediate bf16 cast), or
  producing an output dtype different from `x`'s.
- Any additional pass over the data or global scratch buffer.

## Permitted PyTorch operations

- `torch.empty_like(x)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.numel`, `Tensor.stride`;
  `Tensor.contiguous` / `Tensor.view` as no-op guards on the flat inputs.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
