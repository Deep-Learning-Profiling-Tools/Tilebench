# matrix_copy: canonical algorithm contract

## Functional semantics

Given a square matrix `A` of shape `(N, N)`, produce a new matrix `B` of the
same shape and dtype with `B[i, j] == A[i, j]` for every element. The
reference computes `A + 0`; for every value the benchmark supplies this is an
identity, so a bit-exact copy satisfies the reference (a signed zero copied
bitwise compares equal by value).

The entry point is called as `run(A, N, **kwargs)`. `N` is the side length and
equals `A.shape[0] == A.shape[1]`. Keyword arguments such as `block_size` and
`autotune` may be accepted and ignored.

## Inputs and outputs

- `A`: `(N, N)`, contiguous row-major, dtype one of fp16, bf16, fp32, int8.
  Read-only: `A` must not be written, and the result must not alias it.
- `B` (returned): `(N, N)`, same dtype as `A`, freshly allocated inside
  `run()` on every call, contiguous row-major. Returned as a single tensor.
- `N*N` is not guaranteed to be a multiple of any tile length in general; an
  implementation must not read or write outside the `N*N` elements.

## Required logical stages

1. **Allocate** the output `B` (uninitialised allocation is sufficient).
2. **Stream copy**: read every element of `A` exactly once and write it
   unchanged to the corresponding position of `B` exactly once.

Stage 2 is a single logical stage with no inter-element dependencies. It is
expected to be one launch; splitting the copy across several launches is
permitted but pointless. Nothing may be fused with host-side work, and no
additional passes over the data are allowed.

## Algorithm family and structure

Flat streaming memory copy. No reduction, scan, sort or arithmetic is
involved. The 2-D structure of the matrix carries no semantic weight: because
both `A` and `B` are contiguous with identical layouts, the copy may be
expressed over the flattened `N*N` element range or over 2-D tiles, in any
traversal order, as long as each element is moved exactly once.

## Precision and accumulation

No arithmetic is performed. Values must be moved in the input dtype without
conversion; the output dtype equals the input dtype for all four supported
dtypes. Elements outside the valid range (tail lanes of a partial tile) must
not be stored; masked or padded loads may use any fill value because their
lanes are never written.

## Preprocessing and timing boundary

Everything happens inside `run()` and is timed: the output allocation and the
copy. There is no cast, transpose, packing or cached state of any kind. The
input arrives contiguous and must be consumed as-is; no copy of `A` may be
made before the kernel reads it. Nothing may be cached across calls.

## Permitted implementation mappings

- Tile shape (1-D element ranges or 2-D tiles), elements per program, number
  of programs, vector width, pipelining depth and launch geometry are free.
- Tail handling by explicit masks or by the DSL's bounds-padded loads and
  bounds-clipped stores is free, provided no out-of-range element is written.
- Flattening `A` and `B` to 1-D views on the host is permitted (metadata
  only; it must not copy).
- Whether the lowering uses bulk/asynchronous copies or ordinary loads and
  stores is free.

## Forbidden substitutions

- Returning `A` itself, a view of `A`, or any tensor that shares storage with
  `A`.
- Performing the copy with a PyTorch operation (`A.clone()`, `A.contiguous()`
  on a non-contiguous input, `A + 0`, `A * 1`, `torch.clone`, `B.copy_(A)`,
  `torch.Tensor.to` with `copy=True`, and similar) instead of a kernel.
- Reading or writing the data more than once (for example a staged copy
  through a scratch buffer).
- Changing the dtype at any point.
- Mutating `A`.

## Permitted PyTorch operations

- `torch.empty_like(A)` or `torch.empty(A.shape, dtype=A.dtype, device=A.device)`
  for the output.
- `.view(-1)` / `.reshape(-1)` on `A` and on the output, as metadata-only
  flattening of already-contiguous tensors.
- `A.contiguous()` only as a no-op guard on the already-contiguous input (it
  must not be used as the copy itself).
- Reading dtype, shape, stride and device metadata, and obtaining the current
  stream.

Everything else in `torch` is forbidden inside `run()`.
