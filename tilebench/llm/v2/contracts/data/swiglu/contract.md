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
   dtype, and store, with edge handling where the last tile is partial.

Stage 2 is a single logical stage with no inter-element dependency: each
element of `x` and `y` is read once and each output element written once,
and no intermediate tensor is materialised in global memory. No separate
silu pass followed by a multiply pass, and no global scratch, is permitted,
because materialising `silu(x)` adds an intermediate and a second traversal
and so changes the algorithm rather than its mapping. These are logical
traversal counts, not a guarantee about physical DRAM transactions, which
caches and the compiler may change. How the elements are split across
programs or launches is a mapping choice.

## Algorithm family and structure

Memory-bound elementwise fused gate: two logical reads and one logical write
per element. No reduction, scan or sort. The sigmoid formulation (direct
`1/(1+exp(-x))`, a DSL sigmoid builtin, `exp2` with a `log2(e)` scale, or a
`tanh` identity) is free as long as the result stays within the verifier
tolerance for standard-normal inputs.

## Precision and accumulation

- Both operands are upcast to fp32 before any arithmetic, for all three
  input dtypes.
- The exponential, the add, the reciprocal/division and both multiplies are
  evaluated in fp32.
- A single downcast to `x`'s dtype happens before (or at) the store.
- Tolerances are the verifier's per-dtype defaults (no `verify` override).
- No accumulation.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Flattening views of `x`, `y` and the output are permitted. The inputs are
consumed as given; no copy, cast, cached state or precomputation outside
`run()`.

## Permitted implementation mappings

- Elements per program, the assignment of elements to programs (contiguous
  blocks over the flattened tensors or any other partition), 1-D flat versus
  2-D grid, pipelining, vector width and other launch parameters.
- The number of launches is free: the element range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and no intermediate is written to global memory.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for the last partial block; padded lanes are never stored. Edge handling is
  required wherever the task's fixed shape is not a multiple of the chosen
  tile; supporting shapes other than the task's declared shape is not
  required.
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
- A second logical pass over the data or a global intermediate for
  `silu(x)`, because either adds a traversal the fused gate does not have.
- Writing into `x` or `y` in place or returning a view of an input.

## Permitted PyTorch operations

- `torch.empty_like(...)` or `torch.empty(...)` for the output only.
- `Tensor.contiguous` / `Tensor.view` / `Tensor.reshape` to flatten the
  inputs and to view the output back to `x.shape`; `Tensor.numel`,
  `Tensor.stride`, `.shape`, `.dtype`, `.device`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
