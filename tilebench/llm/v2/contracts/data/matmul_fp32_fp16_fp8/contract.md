# matmul_fp32_fp16_fp8: canonical algorithm contract

## Functional semantics
Dense matrix multiplication C = A @ B with A of shape (M, K) and B of shape
(K, N) in the same dtype; C has shape (M, N) in that dtype. For the fp8 case
the semantics are the unscaled product (unit per-row and per-column
scales), so mathematically C = A @ B for every dtype.

## Inputs and outputs
- A: shape (M, K), row-major contiguous, dtype fp32 | fp16 | fp8_e4m3fn.
  Read-only.
- B: shape (K, N), row-major contiguous, same dtype as A. Read-only. B is
  delivered in this layout; no transposed, column-major or otherwise
  repacked copy is provided.
- C: a freshly allocated (M, N) tensor of A's dtype allocated inside the
  entry point (fp8 output for fp8 inputs). No aliasing.
- The entry point takes (A, B) positionally; no keyword arguments are passed.

## Required logical stages
1. Output allocation.
2. Tiled GEMM: for each logical output tile, loop over K in chunks,
   accumulate the products in a local fp32 accumulator, cast once to C's
   dtype and store the tile.
Stage 2 depends on stage 1. One launch suffices. A split over K with a
deterministic final combine is permitted if its scratch is allocated inside
the entry point on every call.

## Algorithm family and structure
Blocked dense matrix multiplication with fp32 accumulation. The K reduction
order is free. Tails beyond M, N or K must contribute 0 and must never be
stored; the benchmark shapes happen to be tile-aligned, but correctness
must not depend on it.

## Precision and accumulation
- fp32 inputs: operands may be rounded to TF32 (10-bit mantissa) for the
  multiply; accumulation is fp32. Operand formats narrower than TF32 are
  forbidden.
- fp16 inputs: native fp16 operands, fp32 accumulation.
- fp8_e4m3fn inputs: native e4m3 operands fed directly, without scale
  factors, fp32 accumulation.
- Output: a single rounding from the fp32 accumulator to A's dtype,
  including the fp8 output of the fp8 case.

## Preprocessing and timing boundary
The entry point must consume A and B exactly as delivered. Any layout
transform of B (for example forming a K-major copy so that both operands
are read along K) must be performed inside the entry point on every call
and is timed. No cross-call cache keyed by tensor identity, address, shape
or dtype may hold a transformed operand, a descriptor or an output. The
output allocation is timed. No prepacked inputs are provided or may be
assumed.

## Permitted implementation mappings
Logical tile shapes, launch parameters, grouped or swizzled tile ordering,
software pipelining depth, descriptor-based versus pointer-based loads,
reading B directly as (K, N) tiles versus through a per-call in-run K-major
copy, compile-time versus runtime K, persistent scheduling, where the final
cast is applied.

## Forbidden substitutions
torch.matmul / torch.mm / torch.addmm / torch.bmm / torch.einsum /
torch._scaled_mm / the @ operator; caching a transposed or repacked B (or
anything derived from an input) across calls; down-casting fp32 operands
below TF32 or fp16 operands to fp8; applying scale factors in the fp8
case; changing torch.backends matmul precision flags.

## Permitted PyTorch operations
- torch.empty for C and, only if the design uses them, for a per-call
  K-major copy of B or split-K scratch.
- Tensor.t() / Tensor.contiguous() only to produce such a per-call, in-run
  copy (timed, never cached).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.

## Open review items
- Whether a K-major (transposed) repack of B belongs inside the timed
  boundary of this operator, as this contract requires (per call, uncached),
  or whether a prepacked K-major B may be assumed by every implementation.
- Whether the comparison tolerance presumes TF32-class error on the fp32
  verification side at run time, given that process-wide matmul precision
  flags can be altered by other operators in the same process.
