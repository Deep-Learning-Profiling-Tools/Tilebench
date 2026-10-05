# batched_matmul: canonical algorithm contract

## Functional semantics

Batched dense matrix multiplication. For every batch index `b < BATCH`:

```
C[b, m, n] = sum over k < K of A[b, m, k] * B[b, k, n]
```

exactly `torch.matmul(A.view(BATCH, M, K), B.view(BATCH, K, N)).view(-1)`.
The reference enables TF32 for fp32 matrix products.

The entry point is called as `run(A, B, BATCH, M, N, K)`; the four ints are
positional. The entry point is called positionally with exactly the inputs listed below;
no keyword arguments are passed (declared keyword parameters, if any, must
have defaults and must not affect the computation).

## Inputs and outputs

- `A`: flat `(BATCH*M*K,)`, contiguous, logically row-major `(BATCH, M, K)`
  with `K` innermost; dtype fp16, bf16 or fp32. Read-only.
- `B`: flat `(BATCH*K*N,)`, contiguous, logically row-major `(BATCH, K, N)`
  with `N` innermost; same dtype as `A`. Read-only.
- Output: flat `(BATCH*M*N,)`, logically row-major `(BATCH, M, N)`, dtype of
  `A`, freshly allocated inside `run()` on every call and returned as a
  single tensor (a flat view of a freshly allocated 3-D buffer is fine). It
  must not alias `A` or `B`.
- The benchmark uses `N == K == M`, but the implementation must not assume
  it, and `M`, `N`, `K` need not be multiples of any tile extent.

## Required logical stages

1. **Allocate** the output (uninitialised allocation suffices).
2. **Batched GEMM**: one device pass in which every `(batch, M-tile,
   N-tile)` output tile is accumulated over the full `K` range into a local
   fp32 accumulator and stored, cast, exactly once.

Stage 2 is the only device work and is a single launch. There is no
split-K pass followed by a reduction pass, and no separate repacking pass
over `A` or `B` (see the open review item).

## Algorithm family and structure

Classic 2-D output tiling of `BATCH` independent GEMMs. Each output element
is produced by exactly one local accumulator that sums the whole `K` extent
sequentially in chunks; each chunk is a tensor-core (or equivalent) matrix
multiply. No split-K, no atomics, no partial-sum scratch. The batch index
may be a grid axis or folded into a linear tile index; the raster order of
tiles (including grouped or swizzled orderings) is free. Edge tiles must be
handled by zero-filled loads (out-of-range `K` lanes contribute zero) and
clipped or masked stores (no out-of-range element is written); tile-multiple
shapes must not be assumed.

## Precision and accumulation

- Accumulation is fp32 over all of `K` for every input dtype.
- fp16 and bf16 inputs: native products, fp32 accumulation.
- fp32 inputs: the expected precision class is a tensor-core product with
  operands rounded to TF32 and fp32 accumulation, matching the reference's
  TF32 setting; the verification tolerance is sized for it. A full-fp32
  product is permitted. Casting fp32 operands to fp16 or bf16 is forbidden.
- The accumulator is cast to `A`'s dtype at the store. Zero is the only
  fill value.

## Preprocessing and timing boundary

Everything `run()` does is timed, on every call. The provisional rule,
pending the open review item below, is: consume `A` and `B` in their given
row-major layouts through zero-copy views; do not transpose, pack, cast or
copy either operand; do not cache anything derived from input values or
input identity across calls. Host-side descriptor or metadata construction
is permitted. The metric formulas count one read of `A`, one read of `B`
and one write of the output.

## Permitted implementation mappings

- Tile extents along `M`, `N` and `K`, grid shape, raster or grouping order,
  pipelining depth, compile-time versus runtime trip counts.
- Structured block loads (including descriptor-based loads), pointer-based
  loads, or gathers; transposing a loaded tile in registers or tile space
  to suit the matrix-multiply primitive's operand layout.
- Whether the batch index is a grid axis or folded into a linear tile
  index.

## Forbidden substitutions

- `torch.matmul`, `torch.bmm`, `torch.mm`, `torch.einsum`, `torch.baddbmm`,
  the `@` operator on tensors, or any other library GEMM.
- Split-K with a separate reduce pass, or atomic accumulation into the
  output.
- Accumulating in less than fp32, or casting fp32 operands below TF32
  precision.
- Materialised transposed or packed copies of `A` or `B` (provisional, see
  the open review item), and any cache keyed on input tensors that survives
  across calls.
- Writing `A` or `B`; returning a view of `A` or `B`.

## Permitted PyTorch operations

- `torch.empty(BATCH*M*N, dtype=A.dtype, device=A.device)` or
  `torch.empty((BATCH, M, N), ...)` for the output.
- `.view(...)` / `.reshape(...)` on the contiguous tensors (`A` as
  `(BATCH, M, K)`, `B` as `(BATCH, K, N)`, the output between 3-D and flat)
  as metadata-only operations.
- Reading dtype, shape, stride and device metadata, and obtaining the
  current stream.

Everything else in `torch` is forbidden inside `run()`.

## Open review items

- Whether `B` may be consumed through a layout other than the given
  row-major `(BATCH, K, N)`: that is, whether a transposed or repacked copy
  of an operand is permitted at all, and if so whether it must be produced
  inside every timed call or may be prepared once and reused across calls.
  Until decided, the provisional rule above (no repack, no cross-call
  cache) applies.
