# radix_sort: canonical algorithm contract

## Functional semantics

Given a 1-D int32 tensor `input` of `N` keys, return a new int32 tensor of
length `N` containing the same keys in ascending order. Only the sorted
values are returned (no permutation indices), and the result must match the
reference `torch.sort(input).values` exactly (zero tolerance).

The benchmark supplies non-negative keys (all values in `[0, 2^31)`), so
ordering by the unsigned 32-bit pattern coincides with signed ascending
order. An implementation may order by the unsigned pattern directly or
handle the sign bit explicitly; either is acceptable.

The entry point is called as `run(input, N)` with `N == input.numel()`. No
keyword arguments are passed.

## Inputs and outputs

- `input`: `(N,)`, contiguous, dtype int32. Read-only: it must never be
  written, must not be passed to any kernel as a destination, and the result
  must not alias it. Repeated calls must be idempotent.
- Returned value: a single `(N,)` int32 tensor allocated inside `run()`.
  It may be whichever of the run-allocated key buffers holds the final
  result; it must not be `input`, a view of `input`, or shared storage with
  it.
- `N` is the task's fixed key count; keys outside `[0, N)` must never be
  counted or written (edge handling wherever `N` is not a multiple of the
  chosen block size).

## Required logical stages

1. **Working copy and scratch**: copy `input` into a run-allocated key
   buffer (this copy is device work and is counted), allocate a second key
   buffer of the same size, and allocate whatever per-pass scratch the
   counting and scan stages need (for example a histogram of counts per
   (digit value, key block) and scan carry storage).
2. **Digit passes**, least-significant digit first, until the digit
   positions together cover all 32 bits of the key (the digit width, and
   hence the number of passes, is an implementation choice). Each pass
   consists of:
   a. **Count**: partition the current key buffer into blocks; for every
      block, count how many keys fall into each digit value of the current
      digit, and write the counts into the histogram.
   b. **Scan**: an exclusive prefix sum over the histogram, ordered so that
      the result for (digit value `d`, block `b`) equals the number of keys
      with a smaller digit value plus the number of keys with digit value
      `d` in blocks before `b`; this is the global write base for that
      (digit, block) pair.
   c. **Stable scatter**: every key is written into the other key buffer at
      `base[digit, block] + rank`, where `rank` is the number of keys with
      the same digit value that precede it within its block. Ranks follow
      the key order within the block, so equal digits preserve their
      relative order and the pass is a stable partition.
   d. **Swap** the roles of the two key buffers.
3. **Return** the key buffer written by the final pass.

Dependencies: within a pass, scan depends on all counts, and every scatter
depends on the completed scan, so count and scatter of the same pass cannot
share a launch without a device-wide barrier. The scan may be one launch or
several (for example chunk totals, a scan of the totals, and a per-chunk
scan with carry). The count of pass `p + 1` may be fused into the scatter of
pass `p`. Each pass depends on the previous pass's scatter. All 32 bits must
be processed regardless of the key distribution; skipping a pass because
its digit happens to be constant is forbidden, because it would make the
number of passes depend on the data.

## Algorithm family and structure

Least-significant-digit-first counting radix sort with stable per-pass
partitioning, alternating between two key buffers. The digit width (and
hence the number of passes) is a mapping choice: any width or sequence of
widths whose positions cover bits 0 to 31 is permitted, provided every pass
is stable. Counts and offsets are exact integers. Within a block the counts
may be formed by any exact method (separate reductions per digit value,
several counters packed into one wide integer, local histograms); the
histogram memory layout (digit-major or block-major) is free as long as the
scan produces the write bases defined above. The in-block rank may be
computed by an exclusive scan of per-key one-hot flags or any equivalent
exact method.

## Precision and accumulation

Everything is exact integer arithmetic: int32 keys, int32 (or wider) counts
and offsets, and any wider packed counters an implementation chooses. No
floating-point arithmetic touches the keys. The output is int32 and must be
bit-identical to the reference.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

The working copy of `input` and every pass (count, scan, stable scatter),
including any device fill of zero-initialised scratch, are device work and
are counted; the allocations and the Python-level buffer swaps are host
work. Nothing may be cached across calls (no reused histograms, no retained
sorted copies, no prepacked keys), and no sorting or partial sorting may be
done before `run()` is entered. An early return of a copy for `N <= 1` is
permitted.

## Permitted implementation mappings

- Keys per block, launch geometry per stage, vector width and pipelining
  depth are free.
- Digit width and pass count (see above); the sign-bit handling.
- The counting scheme (packed multi-field counters in a 64-bit integer,
  per-digit reductions, local histograms with atomics into a zeroed buffer).
- The scan realisation (multi-launch reduce-then-scan, a single-program scan
  when the histogram is small, a single-pass decoupled scan).
- Gathering the write base per key from the histogram, or loading the few
  bases a block needs as a contiguous tile.
- Fusing the next pass's count into the current scatter.
- Tail handling by explicit masks, zero-padded loads whose padded lanes are
  excluded from counts and ranks, and out-of-range destinations for padded
  lanes that the store drops.
- Specialising on the task's fixed `N` (for example as a compile-time
  constant): edge handling is required wherever the fixed `N` is not a
  multiple of the chosen block, and supporting lengths other than the
  task's declared shape is not required.

## Forbidden substitutions

- Sorting with PyTorch (`torch.sort`, `torch.argsort`, `torch.msort`,
  `torch.topk`, `torch.unique`) or computing the histogram or scan with
  PyTorch (`torch.bincount`, `torch.histc`, `torch.cumsum`).
- Replacing the radix passes with a comparison sort (bitonic or merge
  networks, DSL sort primitives) as the sorting method; a DSL sort primitive
  may not be used to produce the final order.
- Most-significant-digit-first, bucket or sample sort as the family.
- Sorting `input` in place or returning it.
- Data-dependent skipping of digit passes.
- Caching any state across calls.

## Permitted PyTorch operations

- `torch.empty`, `torch.empty_like` for the second key buffer and for
  scratch that is fully overwritten before being read.
- `torch.zeros` / `torch.zeros_like` only for scratch that the chosen
  counting scheme requires to be zero-initialised (for example an
  atomically accumulated histogram); this fill is device work and is
  counted.
- `input.clone()` (or an equivalent device copy kernel) for the working
  copy.
- Python-level swapping of buffer references between passes.
- `.view(dtype)` reinterpretation of a run-allocated buffer (metadata only).
- Reading `numel`, `dtype`, `device`, and obtaining the current stream.

Everything else in `torch` is forbidden inside `run()`.
