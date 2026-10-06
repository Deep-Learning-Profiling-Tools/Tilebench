# layernorm: canonical algorithm contract

## Functional semantics
Layer normalisation over the last dimension with affine parameters. With X
of shape (batch, M, K) viewed as batch*M rows of length K, WEIGHT and BIAS
of shape (K,), and a scalar eps:

    mean = (1/K) * sum_j X[row, j]
    var  = (1/K) * sum_j (X[row, j] - mean)^2          (biased variance)
    Y[row, k] = (X[row, k] - mean) * rsqrt(var + eps) * WEIGHT[k] + BIAS[k]

Statistics are computed in fp32; the result is cast back to X.dtype. Rows
are independent.

## Inputs and outputs
- X: shape (batch, M, K), contiguous row-major, dtype fp16 | bf16 | fp32.
  Read-only.
- WEIGHT, BIAS: shape (K,), contiguous, same dtype as X. Read-only; consumed
  as given (no host-side cast, no hoisting across calls).
- eps: keyword float, default 1e-5 (the benchmark does not pass it).
- Y: a freshly allocated tensor with X's shape and dtype, allocated inside
  the entry point. No aliasing.
- The entry point takes (X, WEIGHT, BIAS) positionally; `eps` keeps its
  default and no keyword arguments are passed.

## Required logical stages
1. Shape bookkeeping and allocation: 2-D (batch*M, K) views of X and Y;
   allocate Y.
2. Row statistics: per row, fp32 reduction(s) over the full row yielding
   the mean and the biased variance, then rstd = rsqrt(var + eps). Any
   numerically reasonable formulation is permitted (sum and sum of squares
   in one pass, a centred second pass, or Welford), provided the
   accumulation is fp32 and the divisor is the true row length K.
3. Row apply: per row and column, (X - mean) * rstd * WEIGHT + BIAS in
   fp32, cast once to X.dtype, store.
Stage 3 depends on stage 2 for the same row only. Stages 2 and 3 are
normally fused in one launch with one program per row (re-reading the row
for stage 3 or keeping it on chip are both permitted). Splitting into two
launches with (batch*M,) fp32 mean/rstd buffers allocated inside the entry
point is permitted.

## Algorithm family and structure
Two-pass row normalisation with an affine epilogue. The reduction order
within a row is free. Masked or zero-padded tail columns must contribute 0
to the sums and must never be stored; WEIGHT/BIAS tail lanes must likewise
never reach a stored element.

## Precision and accumulation
All loads are promoted to fp32; statistics, normalisation and the affine
transform are fp32; a single cast to X.dtype happens at the store. The
variance is biased (divide by K) and eps is added inside the reciprocal
square root. Output dtype equals input dtype.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

The 2-D views are metadata only. WEIGHT and BIAS are consumed as given and
are read by the device work on every call. No cross-call caching of
statistics, parameters or outputs.

## Permitted implementation mappings
Column chunk width, launch parameters, rows per program, compile-time
versus runtime K, the variance formulation, per-chunk versus hoisted
WEIGHT/BIAS loads, fused versus split launches, masked versus padded loads.

## Forbidden substitutions
torch.nn.functional.layer_norm / torch.layer_norm / torch.nn.LayerNorm;
torch.mean, torch.var, Tensor.mean, Tensor.var, torch.rsqrt, torch.sum or
any other PyTorch computation of the statistics, the normalisation or the
affine transform inside the entry point; accumulating statistics below fp32.

## Permitted PyTorch operations
- torch.empty_like / torch.empty for Y and, only in a split design, fp32
  (batch*M,) mean and rstd buffers allocated per call.
- Tensor.reshape / Tensor.view for the 2-D views; Tensor.contiguous() only
  when an input is not already contiguous (it must be a no-op otherwise).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
