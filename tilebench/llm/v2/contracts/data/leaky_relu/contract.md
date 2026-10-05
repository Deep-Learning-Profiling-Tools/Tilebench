# leaky_relu: canonical algorithm contract

## Functional semantics
Elementwise leaky ReLU with the fixed negative slope 0.01:

    Y[i] = X[i]          if X[i] > 0
    Y[i] = 0.01 * X[i]   otherwise

The dtype is preserved. X[i] == 0 yields 0 on either branch, so the strict
comparison is not observable.

## Inputs and outputs
- X: shape (N,), contiguous, dtype fp16 | bf16 | fp32. Read-only.
- N: Python int, the element count (second positional argument).
- Y: a freshly allocated tensor with X's shape and dtype, allocated inside
  the entry point. No aliasing.
- The entry point takes (X, N) positionally plus `block_size`, `autotune`
  and `**kwargs`; extra keywords must be accepted and may be ignored.

## Required logical stages
1. Output allocation.
2. Elementwise map: read X exactly once, write Y exactly once.
Stage 2 depends on stage 1. One logical stage, one launch.

## Algorithm family and structure
Single-pass elementwise map (select between identity and a constant scale).
No reduction, scan or sort. Lanes beyond N must never be stored.

## Precision and accumulation
Compute in the input dtype, or in fp32 with a single cast back to the input
dtype; never in a dtype narrower than the input. The slope is the literal
constant 0.01, not a parameter. Output dtype equals input dtype.

## Preprocessing and timing boundary
Everything inside the entry point is timed: the allocation and the launch.
No casts, copies or layout changes of X; no cross-call caching.

## Permitted implementation mappings
Block length, launch parameters, grid ordering, a where/select versus
max-style arithmetic formulation, whether N is a kernel argument or inferred
from the array bounds, masked versus zero-padded tail handling.

## Forbidden substitutions
torch.nn.functional.leaky_relu, torch.nn.LeakyReLU, torch.nn.functional.relu,
torch.where, torch.maximum, torch.clamp or any other PyTorch elementwise
computation of the result; in-place modification of X; returning X itself.

## Permitted PyTorch operations
- torch.empty_like(X), or torch.empty with X's shape, dtype and device, for
  Y.
- Reads of shape / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
