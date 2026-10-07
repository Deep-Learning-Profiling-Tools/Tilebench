# weight_dequant: canonical algorithm contract

## Functional semantics

Block-wise scale multiplication ("dequantisation" with one scalar scale per
square block of the weight). For `X` of shape `(M, N)` and a scale matrix
`S` of shape `(ceil(M / T), ceil(N / T))`, where `T` is the block side
given by the `TILE_SIZE` argument:

    Y[i, j] = X[i, j] * S[i // T, j // T]        for 0 <= i < M, 0 <= j < N

Floor division maps every element to its block; a partial edge block maps
to the last row/column of `S`, which is why `S` has ceil-sized dimensions.
The reference builds `row_idx = arange(M) // T`, `col_idx = arange(N) // T`,
gathers the full `(M, N)` scale map `S[row_idx[:, None], col_idx[None, :]]`
and returns `X * scale` in the input dtype. That materialised scale map is
reference-side only and must not be reproduced. `X` and `S` share one
floating dtype; `X` is not an integer tensor. There is no other
reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(X, S, M, N, TILE_SIZE)`: the five problem
arguments positional; no keyword arguments are passed.

- `X`: `(M, N)`, contiguous row-major, dtype one of fp16, bf16, fp32.
  Read-only.
- `S`: `(ceil(M / T), ceil(N / T))`, contiguous row-major, same dtype as
  `X`. Read-only.
- `M`, `N`, `TILE_SIZE`: Python ints (every configured case is square, `N == M`; the implementation may specialise per case shape and need not support shapes outside the configured cases).
- Returned: one new tensor `(M, N)` of `X`'s dtype, allocated inside
  `run()` on every call. It must not alias either input.

## Required logical stages

1. **Allocate** the output.
2. **Scaled elementwise map**: for every element, determine its block
   coordinates `(i // T, j // T)`, fetch the corresponding scale from `S`,
   multiply, store (with edge handling wherever a case's shape is
   not a multiple of the chosen tile).

Stage 2 is a single logical stage with no inter-block dependency; how it is
distributed over launches is a mapping choice. No pass that materialises an
expanded `(M, N)` scale map, and no global scratch, is permitted, because
the canonical algorithm derives each scale address on the fly and has no
global-memory intermediate.

## Algorithm family and structure

Memory-bound elementwise multiply with a block-constant broadcast scale: one
logical traversal of `X`, one write of `Y` per element, and a scale lookup
whose address derives from the element's row and column by floor division by
`T`; these are passes in the algorithm, not guarantees about physical DRAM
transactions, which caches, TMA and the compiler may change. No reduction,
scan or sort. Whether the lookup is a per-element gather (flat 1-D
addressing, recovering `row = off // N`, `col = off % N`) or one scale load
per logical tile aligned to the `T x T` blocks (2-D addressing) is a mapping
choice.

## Precision and accumulation

- The multiply is performed in the input dtype, exactly as the reference
  (`X * scale` in the native dtype); computing in fp32 and casting back
  once is accepted within the verifier's per-dtype defaults (no `verify`
  override).
- The output dtype equals `X`'s dtype.
- No accumulation.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Flattening or 2-D views are metadata only. `X` and `S` are consumed as
given; no copy, cast, expansion of `S`, cached state or precomputation
outside `run()`.

## Permitted implementation mappings

- Elements per program, 1-D flat versus 2-D tiled grid (tile-aligned or
  not), number of launches, pipelining, vector width and other launch
  parameters.
- Per-element scale gather versus per-tile scale hoisting.
- Explicit masks or the DSL's bounds-padded loads, gathers and
  bounds-clipped stores for edge tiles; padded lanes are never stored.
- Whether `N`, `T` and the `S` row stride are compile-time constants or
  runtime arguments; recomputing `ceil(N / T)` on the host as the row
  stride of `S` is fine.
- Computing in the input dtype or in fp32.

## Forbidden substitutions

- Materialising the expanded scale map with PyTorch (`torch.arange` index
  construction, advanced indexing of `S`, `repeat_interleave`, `expand`,
  `kron`, `tile`) and multiplying on the host (`X * scale`, `torch.mul`),
  or any other PyTorch arithmetic on the tensors inside `run()`.
- Any scale indexing other than floor division of the row and column by
  `T` (for example rounding, or indexing `S` by the flat offset).
- A second logical traversal of the data or a global-memory intermediate.
- Writing into `X` or `S` in place or returning a view of an input.

## Permitted PyTorch operations

- `torch.empty((M, N), dtype=X.dtype, device=X.device)` for the output
  only.
- `Tensor.contiguous` / `Tensor.view` / `Tensor.reshape` for flat or 2-D
  views of `X`, `S` and the output; `Tensor.numel`, `Tensor.stride`,
  `.shape`, `.dtype`, `.device`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`.
