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
   packed rows of B in chunks; for each chunk load the packed bytes, decode
   each of the four fields on the device (shift, mask, subtract 1), pair
   field i with the A columns i * K_b + r of the same packed rows r, and
   accumulate the int8 x int8 products in a local int32 accumulator; store
   the int32 tile.
Stage 2 depends on stage 1. One launch suffices. The decode must happen on
the device inside the multiplication pass; no unpacked copy of B may be
materialised in global memory, because a separate decode pass would write
and re-read a four times larger unpacked matrix that the canonical
in-mainloop decode never materialises.

## Algorithm family and structure
Blocked integer matrix multiplication with in-mainloop weight decoding. The
visiting order of the logical K index is free (integer accumulation is
exact, so the order cannot change the result). Whether a loaded packed
chunk is reused for all four fields (fields-inner) or the packed rows are
revisited once per field (fields-outer) is an implementation choice, not
part of the canonical algorithm; what is fixed is that the packed B is
decoded correctly on the device inside the GEMM pass and that no unpacked
(K, N) B is written to global memory. Tail handling, wherever the chosen
tile does not divide K_b, N or M of the task's shape: a zero-filled packed
byte decodes to -1, not 0, so packed rows or columns beyond K_b or N must
be masked (or paired only with zero A columns) so that they contribute
nothing; stores are clipped to (M, N). Implementations must be correct for
the task's declared shape (K is a multiple of 1024 in every task of this
operator); supporting shapes other than the declared one, including
arbitrary K_b, is not required.

## Precision and accumulation
Exact integer arithmetic: int8 x int8 products accumulated in int32. No
floating-point evaluation, no TF32, no scaling. Output int32.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

A and the packed B are delivered row-major; no prepacked (K-major) input is
provided or may be assumed. The entry point may consume them exactly as
delivered, or it may form a layout transform of the packed B (for example a
K-major copy of the packed matrix, still packed), provided the transform is
performed by device work inside the entry point on every call: its device
time is counted. Decoding remains inside the multiplication path: a fully
materialised unpacked (K, N) B in global memory is forbidden whether or not
it is rebuilt per call. No cross-call cache keyed by tensor identity,
address, shape, dtype or values may hold a transformed or decoded operand,
a descriptor or an output; every call rebuilds whatever it needs.

## Permitted implementation mappings
Logical tile shapes, launch parameters, grouped tile ordering, pipelining;
fields-inner versus fields-outer loop nesting, and whether a loaded packed
chunk is reused for all four fields or reloaded per field; widening the
packed tile to a wider integer type before masking versus narrower decode
arithmetic; descriptor-based versus pointer loads; compile-time versus
runtime K_b; reading B directly as (K_b, N) tiles versus through a per-call
in-run K-major copy of the packed matrix (device work, counted).

## Forbidden substitutions
torch.matmul / torch.mm / torch._int_mm / torch.addmm / torch.einsum / the
@ operator; materialising an unpacked (K, N) B in global memory, whether by
host-side PyTorch bit operations or by a separate device decode pass;
floating-point evaluation of the product; caching a transposed, repacked or
decoded B (or anything derived from an input's identity, address, shape or
values) across calls, or assuming a prepacked B; reading or changing
process-wide PyTorch precision or backend settings (evaluator state, not
candidate-controlled state).

## Permitted PyTorch operations
- torch.empty for C and, only if the design uses it, for a per-call K-major
  copy of the packed B.
- Tensor.t() / Tensor.contiguous() only to produce such a per-call, in-run
  copy of the packed B (device work inside the entry point on every call,
  counted, never cached).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
