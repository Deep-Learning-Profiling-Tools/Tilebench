# kl_divergence: canonical algorithm contract

## Functional semantics
Row-wise Kullback-Leibler divergence of a target distribution Y against a
prediction LOG_P given as log-probabilities, not reduced over rows. For
inputs of shape (rows, cols):

    LOSS[r] = sum_{c} Y[r, c] * ( log(Y[r, c]) - LOG_P[r, c] )

with the natural logarithm. Convention 0 * log(0) = 0: where Y[r, c] <= 0
the log term is replaced by 0 so that the element contributes 0. (The
benchmark data are softmax outputs and strictly positive, so this guard only
matters for exact zeros, but it must be present.)

## Inputs and outputs
- LOG_P: shape (rows, cols), fp32, row-major with unit column stride (the
  row stride may be read from the tensor). Read-only.
- Y: shape (rows, cols), fp32, same layout. Read-only.
- LOSS: a freshly allocated fp32 tensor of shape (rows,), allocated inside
  the entry point. No aliasing.
- The entry point takes (LOG_P, Y) positionally; no keyword arguments are
  passed.

## Required logical stages
1. Output allocation of LOSS.
2. Row map-reduce: for each row, stream the columns, form the elementwise
   term in fp32, accumulate it over the full row and write one scalar.
Stage 2 depends on stage 1. The elementwise map and the row reduction are
one logical stage: the natural realisation is one program per row in a
single launch. Splitting a row across several programs with a second-level
combine is permitted provided LOSS[r] is still written exactly once and any
partial-sum scratch is allocated inside the entry point.

## Algorithm family and structure
Fused elementwise map plus full-row sum reduction. Every input element is
read exactly once. The accumulation order within a row is free (sequential
column chunks into lane-wise partials followed by a tree, or any other
order); the verification tolerance (config.verify) absorbs ordering
differences. Columns beyond cols (masked or zero-padded tails) must
contribute exactly 0.

## Precision and accumulation
The elementwise term and the accumulator are fp32; LOSS is fp32. The
logarithm is the natural log evaluated at fp32 precision (an approximate
intrinsic within the tolerance is acceptable). The guard for non-positive Y
must be applied before the multiplication so that no NaN or infinity is
produced.

## Preprocessing and timing boundary
Everything inside the entry point is timed: reading the shapes, allocating
LOSS and launching. No host-side casts, copies or layout changes of the
inputs; no cross-call caching of anything.

## Permitted implementation mappings
Column chunk width, launch parameters, number of rows per program, runtime
versus compile-time column count, reducing lane partials per chunk or once
at the end, masked loads versus zero-padded loads, explicit tail masks
versus padding.

## Forbidden substitutions
torch.nn.functional.kl_div / torch.kl_div; torch.log, torch.sum, Tensor.sum,
torch.xlogy or any PyTorch arithmetic computing the term or the row sum;
accumulating in a precision lower than fp32; omitting the non-positive guard.

## Permitted PyTorch operations
- torch.empty for LOSS (shape (rows,), fp32, device of LOG_P) and, only in a
  split-row design, one fp32 partial-sum buffer allocated per call.
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
