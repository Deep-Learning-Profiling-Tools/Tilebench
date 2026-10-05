# matmul_int8: canonical algorithm contract

## Functional semantics
Integer matrix multiplication against a 2-bit-packed weight matrix. A is
int8 of shape (M, K). B is uint8 of shape (K_b, N) with K = 4 * K_b; every
byte B[r, n] holds four 2-bit fields, field i (i = 0..3) occupying bits
2i..2i+1. The decoded weight matrix W of shape (K, N) is

    W[i * K_b + r, n] = ((B[r, n] >> (2 i)) & 3) - 1      in {-1, 0, 1, 2}

and the result is the exact integer product

    C = A @ W,   C[m, n] = sum_i sum_r A[m, i * K_b + r] * W[i * K_b + r, n]

with C of shape (M, N) in int32.

## Inputs and outputs
- A: shape (M, K), int8, row-major contiguous. Read-only.
- B: shape (K_b, N), uint8, row-major contiguous, delivered packed.
  Read-only. No unpacked, transposed or otherwise repacked copy is provided.
- C: a freshly allocated (M, N) int32 tensor allocated inside the entry
  point. No aliasing.
- K is a multiple of 4 (K_b = K / 4). The entry point takes (A, B)
  positionally; no keyword arguments are passed.

## Required logical stages
1. Output allocation.
2. Decode-and-multiply GEMM: for each logical output tile, loop over the
   packed rows of B in chunks; for each chunk load the packed bytes once,
   decode each of the four fields on the device (shift, mask, subtract 1),
   pair field i with the A columns i * K_b + r of the same packed rows r,
   and accumulate the int8 x int8 products in a local int32 accumulator;
   store the int32 tile.
Stage 2 depends on stage 1. One launch suffices. The decode must happen on
the device inside the multiplication pass; no unpacked copy of B may be
materialised in global memory.

## Algorithm family and structure
Blocked integer matrix multiplication with in-mainloop weight decoding.
The visiting order of the logical K index is free (integer accumulation is
exact, so the order cannot change the result). Each packed byte is read
from global memory once per output tile and reused for all four fields.
Tail handling: a zero-filled packed byte decodes to -1, not 0, so packed
rows or columns beyond K_b or N must be masked (or paired only with zero A
columns) so that they contribute nothing; stores are clipped to (M, N).
Implementations must be correct for every benchmark shape (K a multiple of
1024 with M = N fixed); see the open review item on general K.

## Precision and accumulation
Exact integer arithmetic: int8 x int8 products accumulated in int32. No
floating-point evaluation, no TF32, no scaling. Output int32.

## Preprocessing and timing boundary
The entry point must consume A and B exactly as delivered. Any layout
transform of B (for example a K-major copy of the packed matrix) must be
performed inside the entry point on every call and is timed. No cross-call
cache keyed by tensor identity, address, shape or dtype may hold a
transformed or decoded operand. The output allocation is timed. No
prepacked inputs are provided or may be assumed.

## Permitted implementation mappings
Logical tile shapes, launch parameters, grouped tile ordering, pipelining;
fields-inner versus fields-outer loop nesting; widening the packed tile to
a wider integer type before masking versus narrower decode arithmetic;
descriptor-based versus pointer loads; compile-time versus runtime K_b;
reading B directly as (K_b, N) tiles versus through a per-call in-run
K-major copy.

## Forbidden substitutions
torch.matmul / torch.mm / torch._int_mm / torch.addmm / torch.einsum / the
@ operator; host-side decoding of B with PyTorch bit operations into an
unpacked (K, N) matrix; floating-point evaluation of the product; caching a
transposed or decoded B across calls; changing torch.backends matmul flags.

## Permitted PyTorch operations
- torch.empty for C and, only if the design uses it, for a per-call K-major
  copy of the packed B.
- Tensor.t() / Tensor.contiguous() only to produce such a per-call, in-run
  copy (timed, never cached).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.

## Open review items
- Whether a K-major repack of the packed B belongs inside the timed boundary
  of this operator, as this contract requires (per call, uncached), or
  whether a prepacked K-major B may be assumed by every implementation.
- Whether implementations must support arbitrary K_b (tails that are not a
  multiple of the packed-row chunk) or only the benchmark's shapes, given
  that zero-filled packed bytes decode to -1.
- Whether the verification harness must isolate process-wide matrix
  multiplication precision flags so that the verification of other
  operators in the same process is unaffected.
