# l2_norm: canonical algorithm contract

## Functional semantics
L2 normalisation along the last dimension. With X of shape (batch, M, K)
viewed as batch*M rows of length K, and a scalar eps:

    Y[row, k] = X[row, k] * rsqrt( sum_{j < K} X[row, j]^2 + eps )

The sum of squares is formed in fp32, eps is added to the sum of squares
inside the reciprocal square root, and the scaled value is cast back to
X.dtype. Rows are independent.

## Inputs and outputs
- X: shape (batch, M, K), contiguous row-major, dtype fp16 | bf16 | fp32.
  Read-only.
- eps: Python float, second positional argument (default 1e-6).
- Y: a freshly allocated tensor with X's shape and dtype, allocated inside
  the entry point. It may be allocated as a (batch*M, K) buffer and returned
  as a reshaped view of that buffer; it must not alias X.
- The entry point takes (X, eps) positionally plus `autotune` and
  `**kwargs`; extra keywords must be accepted.

## Required logical stages
1. Shape bookkeeping and allocation: view X as (batch*M, K) (a view; a copy
   is permitted only when X is not contiguous) and allocate Y.
2. Row statistics: per row, accumulate the sum of squares over the full
   row in fp32 and form rstd = 1 / sqrt(sumsq + eps) in fp32.
3. Row apply: per row and column, Y = X * rstd in fp32, cast once to
   X.dtype, store.
Stage 3 depends on stage 2 for the same row only. Stages 2 and 3 are
normally fused in one launch with one program per row; re-reading the row
from memory for stage 3 or keeping it on chip are both permitted. Splitting
into two launches with a (batch*M,) fp32 statistics buffer allocated inside
the entry point is permitted.

## Algorithm family and structure
Two-pass row normalisation (statistics, then scale). The reduction order
within a row is free. Masked or zero-padded tail columns must contribute 0 to
the sum of squares and must never be stored.

## Precision and accumulation
Loads are promoted to fp32 before squaring; the accumulator is fp32; rstd is
fp32 (a reciprocal-square-root intrinsic or 1 / sqrt are both acceptable);
the multiply is fp32; a single cast to X.dtype happens at the store. Output
dtype equals input dtype.

## Preprocessing and timing boundary
Everything inside the entry point is timed: the reshape, the allocation and
the launch(es). No cross-call caching of norms, reciprocal norms or
outputs; no precomputed statistics are provided or may be assumed.

## Permitted implementation mappings
Column chunk width, launch parameters, rows per program, compile-time versus
runtime row length, row re-read versus on-chip retention between the two
passes, fused versus two-launch realisation, masked versus padded loads.

## Forbidden substitutions
torch.nn.functional.normalize, torch.linalg.vector_norm / torch.linalg.norm /
Tensor.norm, torch.rsqrt, torch.sqrt, torch.sum, Tensor.square, Tensor.pow
or any other PyTorch computation of the statistics or the scaling applied
to X inside the entry point; accumulating the sum of squares below fp32.

## Permitted PyTorch operations
- torch.empty_like / torch.empty for Y and, only in a two-launch design, one
  fp32 (batch*M,) statistics buffer allocated per call.
- Tensor.reshape / Tensor.view for the 2-D view of X and for returning Y in
  X's shape; Tensor.contiguous() only when X is not already contiguous.
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
