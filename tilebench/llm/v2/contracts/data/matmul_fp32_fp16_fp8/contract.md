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
order is free. Wherever the chosen tile does not divide a case's M, N or
K, tails beyond them must contribute 0 and must never be stored; supporting shapes outside the task's configured cases is not required.

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

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

A and B are delivered row-major; no prepacked input is provided or may be
assumed. The entry point may consume them exactly as delivered, or it may
form a layout transform of B (for example a K-major copy so that both
operands are read along K) in B's own dtype, provided the transform is
performed by device work inside the entry point on every call: its device
time is counted. No cross-call cache keyed by tensor identity, address,
shape, dtype or values may hold a transformed operand, a descriptor or an
output; every call rebuilds whatever it needs. Process-wide PyTorch
precision and backend settings are evaluator state, not candidate state:
the entry point must neither read nor change them (see Forbidden
substitutions); the precision class above is achieved inside the kernel.

## Permitted implementation mappings
Logical tile shapes, launch parameters, grouped or swizzled tile ordering,
software pipelining depth, descriptor-based versus pointer-based loads,
reading B directly as (K, N) tiles versus through a per-call in-run K-major
copy, compile-time versus runtime K, persistent scheduling, where the final
cast is applied.

## Forbidden substitutions
torch.matmul / torch.mm / torch.addmm / torch.bmm / torch.einsum /
torch._scaled_mm / the @ operator; caching a transposed or repacked B (or
anything derived from an input's identity, address, shape or values)
across calls, or assuming a prepacked B; down-casting fp32 operands below
TF32 or fp16 operands to fp8; applying scale factors in the fp8 case;
reading or changing process-wide PyTorch precision or backend settings
(`torch.backends.*`, `torch.set_float32_matmul_precision`, `allow_tf32`):
these are evaluator state, not candidate-controlled state.

## Permitted PyTorch operations
- torch.empty for C and, only if the design uses them, for a per-call
  K-major copy of B or split-K scratch.
- Tensor.t() / Tensor.contiguous() only to produce such a per-call, in-run
  copy (device work inside the entry point on every call, counted, never
  cached).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
