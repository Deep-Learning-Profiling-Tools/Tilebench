# sigmoid: canonical algorithm contract

## Functional semantics

Elementwise logistic function over a 1-D tensor of `N` elements:

    y[i] = 1 / (1 + exp(-x[i]))        for 0 <= i < N

This is the reference `torch.sigmoid(X)`. The integer `N` passed alongside
`X` is the element count (`N == X.numel()` in the benchmark). There is no
reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(X, N)`: `X` and `N` positional; no keyword
arguments are passed.

- `X`: `(N,)`, contiguous, dtype one of fp16, bf16, fp32. Read-only.
- `N`: Python int, the number of elements to process (one of the task's configured sizes).
  Elements at index `>= N` must never be read or written.
- Returned: one new tensor of `X`'s shape and dtype, allocated inside
  `run()` on every call. It must not alias `X`.

## Required logical stages

1. **Allocate** the output.
2. **Elementwise map**: load each element of `X`, upcast to fp32, evaluate
   the sigmoid, cast to the output dtype, and store, with edge handling
   where the last tile is partial.

Stage 2 is a single logical stage with no inter-element dependency: each
element of `X` is read once and each output element written once, and no
intermediate tensor is materialised in global memory. No second pass over
the data is permitted, because it would change the algorithm rather than its
mapping. These are logical traversal counts, not a guarantee about physical
DRAM transactions, which caches and the compiler may change. How the
elements are split across programs or launches is a mapping choice.

## Algorithm family and structure

Memory-bound 1-D elementwise map: one logical read and one logical write per
element. No reduction, scan or sort. The formulation of the
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

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

`X` is consumed as given (a flat contiguous vector); no copy, cast, reshape
beyond a free view, cached state or precomputation outside `run()`.

## Permitted implementation mappings

- Elements per program, the assignment of elements to programs (contiguous
  blocks or any other partition), grid shape (1-D or multi-dimensional),
  pipelining, vector width and other launch parameters.
- The number of launches is free: the index range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and no intermediate is written to global memory.
- Explicit masks or the DSL's bounds-padded loads and bounds-clipped stores
  for the last partial block; the fill value of padded lanes is irrelevant
  because they are never stored. Edge handling is required wherever a case's `N` is not a multiple of the chosen tile; supporting shapes outside the task's configured cases is not required.
- The exact sigmoid formulation (see above).

## Forbidden substitutions

- Calling `torch.sigmoid`, `torch.nn.functional.sigmoid`, `X.sigmoid()`,
  `torch.special.expit`, `nn.Sigmoid`, or computing the result with
  PyTorch elementwise arithmetic (`torch.exp`, `1 / (1 + ...)`) on the host.
- Evaluating the sigmoid in fp16/bf16 for half-precision inputs.
- Processing fewer or more than `N` elements, or reading past `N`.
- Writing into `X` in place or returning a view of `X`.
- Any additional logical pass over the data or global scratch buffer,
  because either changes the algorithm's data movement rather than its
  mapping.

## Permitted PyTorch operations

- `torch.empty_like(X)` or `torch.empty(...)` for the output only.
- Reading `.shape`, `.dtype`, `.device`, `Tensor.numel`, `Tensor.stride`;
  `Tensor.contiguous` / `Tensor.view` as no-op guards on the flat input.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
