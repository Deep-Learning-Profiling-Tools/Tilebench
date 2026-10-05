# sigmoid: canonical algorithm contract

## Functional semantics

Elementwise logistic function over a 1-D tensor of `N` elements:

    y[i] = 1 / (1 + exp(-x[i]))        for 0 <= i < N

This is the reference `torch.sigmoid(X)`. The integer `N` passed alongside
`X` is the element count (`N == X.numel()` in the benchmark). There is no
reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(X, N, block_size=..., autotune=False,
**kwargs)`: `X` and `N` positional; `block_size` carries an integer default
the implementation may ignore, and unknown keywords must be accepted.

- `X`: `(N,)`, contiguous, dtype one of fp16, bf16, fp32. Read-only.
- `N`: Python int, the number of elements to process; it sizes the grid and
  the bounds mask. Elements at index `>= N` must never be read or written.
- Returned: one new tensor of `X`'s shape and dtype, allocated inside
  `run()` on every call. It must not alias `X`.

## Required logical stages

1. **Allocate** the output.
2. **Elementwise map**: load a contiguous block of `X`, upcast to fp32,
   evaluate the sigmoid, cast to the output dtype, store under a bounds
   mask.

Stage 2 is a single logical stage with no inter-block dependency and is
expected to be one launch. No second pass over the data is permitted.

## Algorithm family and structure

Memory-bound 1-D elementwise map with flat contiguous blocking: one read and
one write per element. No reduction, scan or sort. The formulation of the
sigmoid (direct `1/(1+exp(-x))`, a DSL sigmoid builtin, `exp2` with a
log2(e) scale, a `tanh` identity, or a sign-split for numerical range) is
free as long as the result stays within the verifier tolerance for every
input value produced by a standard normal distribution.

## Precision and accumulation

- The sigmoid is evaluated in fp32 after an explicit upcast of the loaded
  element, for all three input dtypes.
- A single downcast to the input dtype happens before (or at) the store.
- Tolerances are the verifier's per-dtype defaults (no `verify` override).
  Approximate exponential lowering is accepted within them.
- No accumulation.

## Preprocessing and timing boundary

Everything `run()` does is timed: the output allocation and the launch.
`X` is consumed as given (a flat contiguous vector); no copy, cast, reshape
beyond a free view, cached state or precomputation outside `run()`.

## Permitted implementation mappings

- Elements per program, grid shape (1-D or multi-dimensional), pipelining,
  vector width and other launch parameters.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for the last partial block; the fill value of padded lanes is irrelevant
  because they are never stored.
- The exact sigmoid formulation (see above).

## Forbidden substitutions

- Calling `torch.sigmoid`, `torch.nn.functional.sigmoid`, `X.sigmoid()`,
  `torch.special.expit`, `nn.Sigmoid`, or computing the result with
  PyTorch elementwise arithmetic (`torch.exp`, `1 / (1 + ...)`) on the host.
- Evaluating the sigmoid in fp16/bf16 for half-precision inputs.
- Processing fewer or more than `N` elements, or reading past `N`.
- Writing into `X` in place or returning a view of `X`.
- Any additional pass over the data or global scratch buffer.

## Permitted PyTorch operations

- `torch.empty_like(X)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.numel`, `Tensor.stride`;
  `Tensor.contiguous` / `Tensor.view` as no-op guards on the flat input.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
