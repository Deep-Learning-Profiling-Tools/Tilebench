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
- The entry point takes (X, N) positionally; no keyword arguments are passed.

## Required logical stages
1. Output allocation.
2. Elementwise map: one logical traversal that reads each element of X once
   and writes each element of Y once.
Stage 2 depends on stage 1. One logical stage: no intermediate tensor is
materialised in global memory and there is no second pass over the data,
because either would change the algorithm rather than its mapping. These are
logical traversal counts, not a guarantee about physical DRAM transactions,
which caches and the compiler may change. How the index range is split
across programs or launches is a mapping choice.

## Algorithm family and structure
Single-pass elementwise map (select between identity and a constant scale).
No reduction, scan or sort. Lanes beyond N must never be stored.

## Precision and accumulation
Compute in the input dtype, or in fp32 with a single cast back to the input
dtype; never in a dtype narrower than the input. The slope is the literal
constant 0.01, not a parameter. Output dtype equals input dtype.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

No casts, copies or layout changes of X; no cross-call caching.

## Permitted implementation mappings
Block length, launch parameters, grid ordering, the number of launches the
index range is partitioned into (each element still processed once, no
intermediate written to global memory), a where/select versus max-style
arithmetic formulation, whether N is a kernel argument or inferred from the
array bounds, masked versus zero-padded tail handling. Edge handling is
required wherever a case's N is not a multiple of the chosen tile;
supporting shapes outside the task's configured cases is not required.

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
