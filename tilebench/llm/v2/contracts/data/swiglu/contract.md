# swiglu: canonical algorithm contract

## Functional semantics

Elementwise SwiGLU gate of two tensors of identical shape `(M, N)`:

    out[i, j] = silu(x[i, j]) * y[i, j] = x[i, j] * sigmoid(x[i, j]) * y[i, j]

with `sigmoid(v) = 1 / (1 + exp(-v))`. This is the reference
`torch.nn.functional.silu(x) * y`: `x` is the gated operand, `y` the
multiplier. There is no reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(x, y)`: `x`, `y` positional; no keyword
arguments are passed.

- `x`: `(M, N)`, contiguous row-major, dtype one of fp16, bf16, fp32.
  Read-only.
- `y`: `(M, N)`, contiguous, same dtype as `x`. Read-only. Shape equality
  with `x` may be asserted.
- Returned: one new tensor of `x`'s shape and dtype, allocated inside
  `run()` on every call (a flat buffer viewed back to `(M, N)` is fine). It
  must not alias either input.

## Required logical stages

1. **Allocate** the output.
2. **Elementwise map** over all `M*N` elements: load `x` and `y`, upcast
   both to fp32, compute `x * sigmoid(x) * y`, cast once to the output
   dtype, store under a bounds mask.

Stage 2 is a single logical stage with no inter-block dependency and is
expected to be one launch. No separate silu pass followed by a multiply
pass, and no global scratch, is permitted.

## Algorithm family and structure

Memory-bound elementwise fused gate with flat contiguous blocking over the
flattened tensors: two reads and one write per element. No reduction, scan
or sort. The sigmoid formulation (direct `1/(1+exp(-x))`, a DSL sigmoid
builtin, `exp2` with a `log2(e)` scale, or a `tanh` identity) is free as
long as the result stays within the verifier tolerance for standard-normal
inputs.

## Precision and accumulation

- Both operands are upcast to fp32 before any arithmetic, for all three
  input dtypes.
- The exponential, the add, the reciprocal/division and both multiplies are
  evaluated in fp32.
- A single downcast to `x`'s dtype happens before (or at) the store.
- Tolerances are the verifier's per-dtype defaults (no `verify` override).
- No accumulation.

## Preprocessing and timing boundary

Everything `run()` does is timed: flattening views of `x` and `y`, the
output allocation and the launch. The inputs are consumed as given; no copy,
cast, cached state or precomputation outside `run()`.

## Permitted implementation mappings

- Elements per program, 1-D flat versus 2-D grid, pipelining, vector width
  and other launch parameters.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for the last partial block; padded lanes are never stored.
- The exact sigmoid formulation (see above).
- Whether the element count is a compile-time constant or runtime argument.

## Forbidden substitutions

- Calling `torch.nn.functional.silu`, `F.silu`, `nn.SiLU`,
  `torch.sigmoid`, `F.sigmoid`, `x.sigmoid()`, `torch.special.expit`, a
  library SwiGLU/gated-MLP routine, or computing the gate with PyTorch
  elementwise arithmetic on the host.
- Evaluating the sigmoid or the products in fp16/bf16 for half-precision
  inputs.
- Gating `y` instead of `x` (`y * sigmoid(y) * x`), or applying any other
  activation.
- A second pass over the data or a global intermediate for `silu(x)`.
- Writing into `x` or `y` in place or returning a view of an input.

## Permitted PyTorch operations

- `torch.empty_like(...)` or `torch.empty(...)` for the output only.
- `Tensor.contiguous` / `Tensor.view` / `Tensor.reshape` to flatten the
  inputs and to view the output back to `x.shape`; `Tensor.numel`,
  `Tensor.stride`, `.shape`, `.dtype`, `.device`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
