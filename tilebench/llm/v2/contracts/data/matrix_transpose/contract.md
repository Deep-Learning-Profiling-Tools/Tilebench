# matrix_transpose: canonical algorithm contract

## Functional semantics

Given a 2-D matrix `x` of shape `(m, n)`, produce a new contiguous row-major
matrix `out` of shape `(n, m)` with `out[j, i] == x[i, j]` for every element.
This is the reference `x.transpose(0, 1).contiguous()`: an out-of-place,
materialised transpose, not a strided view.

The entry point is called as `run(x, **kwargs)`; `m` and `n` are read from
`x.shape`. Keyword arguments such as `block_size` and `autotune` may be
accepted and ignored. Input that is not 2-D is outside the benchmark.

## Inputs and outputs

- `x`: `(m, n)`, contiguous row-major, dtype one of fp16, bf16, fp32, int8.
  Read-only: `x` must not be written, and the result must not alias it.
- `out` (returned): `(n, m)`, same dtype as `x`, freshly allocated inside
  `run()` on every call, contiguous row-major (`out.stride() == (m, 1)`).
  Returned as a single tensor.
- `m` and `n` need not be multiples of any tile size in general; no element
  outside `[0, m) x [0, n)` may be read, and no element outside
  `[0, n) x [0, m)` of `out` may be written.

## Required logical stages

1. **Allocate** the `(n, m)` output (uninitialised allocation is sufficient).
2. **Tiled transpose**: read every element of `x` exactly once and write it
   exactly once to its transposed position in `out`.

Stage 2 is a single logical stage with no inter-tile dependencies and is
expected to be one launch. No additional pass over the data is permitted (for
example a first pass producing a strided intermediate followed by a
compaction pass).

## Algorithm family and structure

Out-of-place tiled transpose: the matrix is partitioned into rectangular
logical tiles; each tile is read with the input's row-major addressing and
written with the output's row-major addressing at swapped coordinates. No
reduction, scan or sort is involved. The on-chip re-layout between a
coalesced read and a coalesced write (through registers, local memory, or a
DSL transpose primitive) is a mapping choice. Grid orientation (which input
axis the first grid dimension walks) is free.

## Precision and accumulation

No arithmetic is performed. Values are moved bit-exactly in the input dtype;
output dtype equals input dtype for all four supported dtypes. Lanes of a
partial edge tile that fall outside the matrix must never be stored; the fill
value of masked or padded loads is irrelevant because those lanes are not
written.

## Preprocessing and timing boundary

Everything happens inside `run()` and is timed: the output allocation and the
transpose kernel. There is no cast, packing, host-side copy or cached state.
The input must be consumed in its given layout; it may not be copied, padded
or re-laid-out before the kernel reads it. Nothing may be cached across
calls.

## Permitted implementation mappings

- Tile shape (square or rectangular), elements per program, grid shape and
  traversal order, pipelining depth and vector width are free.
- Reading strided and writing coalesced, reading coalesced and writing
  strided, or re-laying out the tile on chip so that both sides are
  coalesced are all acceptable.
- Edge tiles may be handled by explicit masks or by the DSL's bounds-padded
  loads and bounds-clipped stores.
- Passing explicit strides of `x` and `out` to the kernel is permitted; so is
  assuming the contiguous layouts stated above.

## Forbidden substitutions

- Returning a view (`x.t()`, `x.T`, `x.mT`, `x.transpose(0, 1)`,
  `x.permute(1, 0)`, `torch.transpose`) or any tensor sharing storage with
  `x`.
- Materialising the transpose with PyTorch (`x.t().contiguous()`,
  `x.t().clone()`, `out.copy_(x.t())`, `torch.as_strided` followed by a copy,
  and similar) instead of a kernel.
- Multiple passes over the data or scratch buffers in global memory.
- Any dtype conversion.
- Mutating `x`.

## Permitted PyTorch operations

- `torch.empty((n, m), dtype=x.dtype, device=x.device)` for the output.
- Reading `x.shape`, `x.stride()`, `x.dtype`, `x.device`, `out.stride()`,
  and obtaining the current stream.
- `x.contiguous()` only as a no-op guard on the already-contiguous input.

Everything else in `torch` is forbidden inside `run()`. In particular no
transpose, permute, view-based re-layout, clone, copy or conversion of `x`.
