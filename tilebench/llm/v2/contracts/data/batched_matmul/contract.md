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
  with `N` innermost; same dtype as `A`. Read-only. `B` is delivered in
  this layout; no transposed, column-major or otherwise repacked copy is
  provided.
- Output: flat `(BATCH*M*N,)`, logically row-major `(BATCH, M, N)`, dtype of
  `A`, freshly allocated inside `run()` on every call and returned as a
  single tensor (a flat view of a freshly allocated 3-D buffer is fine). It
  must not alias `A` or `B`.
- Every configured case has `N == K == M`; supporting shapes outside the task's configured cases is not required. Edge handling is required wherever a case's shape is not a multiple of the chosen tile.

## Required logical stages

1. **Allocate** the output (uninitialised allocation suffices).
2. **Batched GEMM**: one logical pass in which every `(batch, M-tile,
   N-tile)` output tile is accumulated over the full `K` range into a local
   fp32 accumulator and stored, cast, exactly once.

Stage 2 is the only required device work. There is no split-K pass followed
by a reduction pass (partial sums would round-trip through global memory
instead of staying in one on-chip accumulator). An implementation may
additionally repack or transpose `A` or `B` into another layout, but only as
device work performed inside every `run()` call (see "Preprocessing and
timing boundary"); such a copy is an optional, counted layout step, not a
change of algorithm. Spreading the output tiles over more than one launch
(for example one launch per batch slice) without any partial-sum
intermediate is a mapping choice.

## Algorithm family and structure

Classic 2-D output tiling of `BATCH` independent GEMMs. Each output element
is produced by exactly one local accumulator that sums the whole `K` extent
sequentially in chunks; each chunk is a tensor-core (or equivalent) matrix
multiply. No split-K, no atomics, no partial-sum scratch, because splitting
`K` would combine partial sums through global memory, a different reduction
structure. The batch index may be a grid axis or folded into a linear tile
index; the raster order of tiles (including grouped or swizzled orderings)
is free. Wherever the chosen tile does not divide a case's `M`, `N` or
`K`, edge tiles must be handled by zero-filled loads (out-of-range `K` lanes
contribute zero) and clipped or masked stores (no out-of-range element is
written); supporting shapes outside the task's configured cases is not required.

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

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

`A` and `B` are delivered in the row-major layouts declared above. An
implementation may consume them directly through zero-copy views, or it may
materialise a repacked or transposed representation of an operand (for
example `B` as `(BATCH, N, K)`) in the operand's own dtype, provided that
copy is produced by device work inside every `run()` call: its device time
is part of the measured quantity, exactly like the GEMM launch. Nothing
derived from an input's values, identity, data pointer or shape may be kept
across calls: no cross-call cache of a transformed operand, descriptor or
output, and no assumption that an input arrives prepacked. Every call
rebuilds whatever it needs. Host-side descriptor or metadata construction
is permitted. The metric formulas count one read of `A`, one read of `B`
and one write of the output; the traffic of a per-call repack is not
credited by them.

## Permitted implementation mappings

- Tile extents along `M`, `N` and `K`, grid shape, raster or grouping order,
  pipelining depth, compile-time versus runtime trip counts.
- Structured block loads (including descriptor-based loads), pointer-based
  loads, or gathers; transposing a loaded tile in registers or tile space
  to suit the matrix-multiply primitive's operand layout.
- Whether the batch index is a grid axis or folded into a linear tile
  index.
- Consuming `A` and `B` in their delivered row-major layouts versus through
  a repacked or transposed copy of an operand rebuilt by device work inside
  every `run()` call (its time counted).
- The number of launches over which the output tiles are spread (for
  example one launch per batch slice), provided no partial-sum intermediate
  is written to global memory.

## Forbidden substitutions

- `torch.matmul`, `torch.bmm`, `torch.mm`, `torch.einsum`, `torch.baddbmm`,
  the `@` operator on tensors, or any other library GEMM.
- Split-K with a separate reduce pass, or atomic accumulation into the
  output (partial sums would round-trip through global memory instead of
  staying in one on-chip accumulator).
- Accumulating in less than fp32, or casting fp32 operands below TF32
  precision.
- Any cache that survives across calls and is keyed on an input's identity,
  data pointer, shape or values (a transposed or repacked operand, a
  descriptor or an output prepared once and reused), and any assumption
  that an operand arrives prepacked. A repacked or transposed copy of `A`
  or `B` is permitted only as device work inside every `run()` call.
- Writing `A` or `B`; returning a view of `A` or `B`.

## Permitted PyTorch operations

- `torch.empty(BATCH*M*N, dtype=A.dtype, device=A.device)` or
  `torch.empty((BATCH, M, N), ...)` for the output.
- `.view(...)` / `.reshape(...)` on the contiguous tensors (`A` as
  `(BATCH, M, K)`, `B` as `(BATCH, K, N)`, the output between 3-D and flat)
  as metadata-only operations.
- `.transpose(...)` / `.permute(...)` / `.t()` followed by `.contiguous()`,
  or `torch.empty` plus a device copy, only to produce a per-call, in-run
  repacked copy of an operand in its own dtype (device work, counted, never
  cached).
- Reading dtype, shape, stride and device metadata, and obtaining the
  current stream.

Everything else in `torch` is forbidden inside `run()`.
