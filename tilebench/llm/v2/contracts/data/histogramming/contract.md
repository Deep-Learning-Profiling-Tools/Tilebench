# histogramming: canonical algorithm contract

## Functional semantics
Integer histogram with identity binning. Given a 1-D int32 vector `input`
of length N and an integer `num_bins`:

    hist[b] = number of indices i with input[i] == b,   for b in [0, num_bins)

i.e. `torch.bincount(input.to(torch.int64), minlength=num_bins).to(torch.int32)`.
The value IS the bin index: no scaling, offset or bin width. Every input
value in the benchmark lies in [0, num_bins); values outside that range must
not corrupt in-range counts (dropping them is acceptable). Counts are exact
int32 integers and verification is exact (zero tolerance).

## Inputs and outputs
- `input`: (N,), int32, contiguous, on the device.
- `N`: Python int equal to input.shape[0]; `num_bins`: Python int >= 1.
- Output: exactly one tensor of shape (num_bins,) in int32, freshly
  allocated inside run() on every call; no aliasing with any input.
- No input may be modified.
- Call form: `run(input, N, num_bins)`, positional; no keyword arguments are
  passed. Never run a configuration search.

## Required logical stages
1. Scratch initialisation: inside run(), allocate a global int32 scratch of
   P private partial histograms, shape (P, num_bins), and zero-initialise
   it. The zero fill may be a device fill issued by run() or done by each
   owning program before it counts. P is a configuration choice for the
   case's input length (see the permitted mappings), and the scratch
   is never kept across calls.
2. Private counting: partition the input among P programs; each program
   owns exactly one private row of the scratch and counts the values of its
   share of the input into that row (bins outside [0, num_bins) are
   skipped). No two programs ever write the same row, so cross-program
   contention does not exist; contention is confined to lanes of one
   program hitting the same bin.
3. Cross-partial reduction: for each bin, sum the P private rows into the
   output (exact int32).

Stage 2 depends on stage 1 (the row must be zero before accumulation);
stage 3 must not start before every program of stage 2 has finished, which
in practice means a separate launch (a fused single launch is permitted
only if the programming model guarantees a grid-wide barrier between the
two stages). Stage 1 may be fused into stage 2 as described.

## Algorithm family and structure
Two-level privatised histogram: order-free exact integer counting into
per-program private partial histograms held in a global scratch, followed by
a column-wise sum over the partials. Stage 2 makes a single logical
traversal of the input in chunks (grid-stride or contiguous partitioning);
this counts passes in the algorithm, not physical memory transactions. Stage
3 is a sequential or tree sum over the P rows per bin slice (order
irrelevant for int32). No sort, no scan, no prefix structure. A single
shared histogram updated by all programs (no privatisation) is a different
algorithm and is not permitted, because it replaces private partial counts
plus a reduction with contended updates of one global histogram.

## Precision and accumulation
- int32 everywhere: the input values, the private counts, the reduction
  accumulator and the output.
- No floating-point conversion at any stage; results must be bit-exact.
- Tolerance: the operator config's verify section (exact).

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Besides host work (the input assertions, the output and scratch allocation
calls, host arithmetic for P and the chunking), run() issues only the
scratch zero fill of stage 1 and the counting and reduction launches. The
zero fill, whether a device fill issued by run() or done in-kernel by the
owning program, is device work and is counted. `.contiguous()` on the input
is permitted only as a no-op. No host-side sort, cast, copy or compaction of
the input; no scratch or partial result cached across calls; nothing
precomputed outside run().

## Permitted implementation mappings
- The number of partials P and the chunk size; grid-stride versus contiguous
  partitioning of the input across programs.
- How a program updates its private row: atomic increments per element
  directly into the row (masked, or adding 0 for invalid lanes), or on-chip
  pre-aggregation of a chunk followed by additive updates of the row.
- Reduce tiling (rows per step, bins per program) and the reduction tree
  within a tile.
- Zero fill by a device fill call versus by the owning program.
- Masked tail handling versus zero-padded loads with excluded lanes.

## Forbidden substitutions
- torch.bincount, torch.histc, torch.histogram, torch.unique / unique_consecutive
  counting, torch.sort/argsort-based counting, torch.scatter_add /
  index_add on the host, or any host-side tensor arithmetic producing the
  counts.
- A single global histogram shared by all programs (no privatisation).
- Floating-point counting or accumulation; any conversion of the input.
- Reusing or caching the scratch between calls; allocating the scratch
  outside run().
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the (num_bins,) output.
- `torch.zeros` for the (P, num_bins) int32 scratch (or `torch.empty` plus
  an in-kernel zero fill by the owning program).
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`, `.stride()`,
  `.is_cuda`; `Tensor.contiguous()` only as a no-op.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
