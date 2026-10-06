# interleave: canonical algorithm contract

## Functional semantics
Given two one-dimensional tensors A and B of equal length N, produce a
one-dimensional tensor OUT of length 2N with

    OUT[2i] = A[i],  OUT[2i + 1] = B[i]   for 0 <= i < N.

This is pure data movement: no arithmetic is performed, every value is copied
bit-exactly and the dtype is preserved (including the integer dtype).

## Inputs and outputs
- A: shape (N,), contiguous, dtype one of fp16, bf16, fp32, int8. Read-only.
- B: shape (N,), same dtype and layout as A. Read-only.
- N: Python int, the element count of A and B (third positional argument).
- OUT: a freshly allocated contiguous tensor of shape (2N,) and dtype A.dtype,
  allocated inside the entry point on every call. It aliases nothing.
- Neither input may be mutated. The entry point takes (A, B, N) positionally;
  no keyword arguments are passed.

## Required logical stages
1. Output allocation: an uninitialised buffer of length 2N (inside the entry
   point, on every call).
2. Interleaved copy: in one logical traversal, read each element of A and B
   once and write each element of OUT once, placing A on even and B on odd
   positions.
Stage 2 depends only on stage 1. There is no further stage, no reduction and
no intermediate buffer. Stage 2 is a single logical pass; how it is
distributed over launches (for example one launch per input, each writing
its stride-2 half of OUT) is a mapping choice, since that adds no
intermediate and no second traversal of either input. Pass counts are
logical traversals in the algorithm, not guarantees about physical DRAM
transactions, which caches, TMA and the compiler may change.

## Algorithm family and structure
Elementwise zip of two streams in one logical traversal (how that traversal
is distributed over launches is a mapping choice, see the stages). For a
logical block of consecutive input indices the corresponding output block is
contiguous and twice as long; whether the zip is formed in registers (stack
the two input blocks and flatten) followed by one contiguous store, or
written as two stride-2 stores, is an implementation choice. Lanes beyond N
(the last block) must never be stored and padded input values must never
reach OUT.

## Precision and accumulation
None. No cast of any kind is permitted; the same code path must serve the
integer dtype.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

No cross-call caching of inputs, outputs or derived buffers; no prepacked or
pre-interleaved inputs exist or may be assumed. Each call returns a new
tensor.

## Permitted implementation mappings
Block length per program, grid shape and ordering, launch parameters, number
of launches (for example one per input), vector width, whether N is passed
to the kernel or inferred from the array bounds, in-register zip versus two
strided stores, masked versus clipped tail handling.

## Forbidden substitutions
Strided slice assignment executed by PyTorch (OUT[0::2] = A, OUT[1::2] = B);
torch.stack / torch.cat / torch.repeat_interleave or any reshape-based
interleave computed by PyTorch; any PyTorch copy of A or B; more than one
logical traversal of the inputs; returning or reusing an output from a
previous call.

## Permitted PyTorch operations
- torch.empty for OUT (dtype and device taken from A).
- Reads of shape / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
