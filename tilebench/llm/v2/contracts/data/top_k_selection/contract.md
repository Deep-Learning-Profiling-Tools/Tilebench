# top_k_selection: canonical algorithm contract

## Functional semantics

Return the `k` largest values of a 1-D fp32 tensor `input` of length `N`,
sorted in descending order, values only (no indices). This is the reference
`torch.topk(input.contiguous(), k, largest=True, sorted=True).values`.
Multiset semantics: if a value occurs several times among the `k` largest,
it appears that many times in the result. `1 <= k <= N`. There is no
reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(input, N, k)`: `input`, `N`, `k` positional;
no keyword arguments are passed.

- `input`: `(N,)`, contiguous, fp32 only (the only dtype swept), on the
  CUDA device. Read-only. `N == input.shape[0]`.
- `N`, `k`: Python ints.
- Returned: a 1-D fp32 tensor of shape exactly `(k,)` in descending order.
  It may be a view (slice) of a buffer allocated inside `run()`; it must not
  be a view of `input` and must not share storage with it.

## Required logical stages

Let `K'` be the smallest power of two that is `>= k`, and let `B` be the
block width, with `B >= 2 * K'` (it need not be a power of two). `K'` is
part of the algorithm, not a tuning choice: it fixes the per-block candidate
width and with it the candidate traffic between levels (a width below `k`
would break exactness; any other width changes that traffic).

1. **Level 0, block selection**: the input is cut into `ceil(N / B)` blocks
   of at most `B` elements (contiguous blocks in address order are the
   natural partition; any fixed partition is permitted). Each block, padded
   with `-inf` beyond `N`, yields its `K'` largest values in descending
   order, written as one row of a fresh `(ceil(N / B), K')` fp32 candidate
   buffer allocated inside `run()`.
2. **Higher levels**: while the previous level produced more than one
   block, view its candidate buffer as a flat vector of `nb * K'`
   candidates and apply the same block selection to it, producing a new
   `(ceil(nb * K' / B), K')` buffer. Because `B >= 2 K'`, every level at
   least halves the candidate count, so the recursion terminates with a
   single block.
3. **Result**: the final level's single row holds the global top `K'` in
   descending order; return its first `k` entries.

Each level needs all candidates of the previous level, so consecutive
levels are separated by a device-wide dependency and each level reads the
previous level's candidates from that level's global buffer. One launch per
level is the natural realisation; how the dependency is enforced is a
mapping choice. When `N <= B` the hierarchy is a single level.

## Algorithm family and structure

Hierarchical (tournament) block top-`K'` selection: leaves are the input
blocks; every internal node is the top `K'` of the concatenation of its
children's top-`K'` lists; the fan-in per level is `B / K' >= 2`. Exactness
follows from `K' >= k`: the `k` largest of any set lie within the `K'`
largest of each block. Within a block, how the top `K'` is obtained (a DSL
block top-k primitive, a full bitonic sort followed by taking the prefix, or
another exact in-block selection network) is a mapping choice, provided it
is exact, emits the values in descending order, keeps duplicates with their
multiplicity, and treats `-inf` padding so that it can never displace a real
value (a genuine `-inf` input is indistinguishable from padding, which is
harmless for values-only output). Comparisons only: no arithmetic on the
values. NaN inputs are outside the benchmark.

## Precision and accumulation

- Comparison-only selection on fp32 values; every returned value is bitwise
  one of the input values.
- Intermediate candidate buffers are fp32.
- Tolerance is `atol 1e-5, rtol 1e-5` (config `verify`), which an exact
  selection meets trivially.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Every level's launch is device work and is counted; computing `K'` and `B`,
the per-level candidate buffer allocations and the final slice are host
work. `input` is consumed as given; no copy, sort, cast, cached state or
precomputation outside `run()`.

## Permitted implementation mappings

- The block width `B` (subject to `B >= 2 K'`; not necessarily a power of
  two), the partition of each level's vector into blocks, elements per
  program, launch parameters, pipelining, and how the dependency between
  levels is enforced.
- The in-block selection primitive (see above); a special case for
  `K' == 1` (block maximum) is fine.
- Allocating a fresh buffer per level or reusing one scratch buffer
  between levels; returning a slice of the final buffer or copying the `k`
  values into a separate `(k,)` tensor (the slice is metadata only; a copy
  is device work and is counted).
- Whether `K'` and `B` are compile-time constants.

## Forbidden substitutions

- `torch.topk`, `torch.sort`, `torch.argsort`, `torch.kthvalue`,
  `torch.msort`, `torch.max`/`torch.amax`, or any host/library selection or
  sort over the data; moving the data to the host (`.cpu()`, `.tolist()`,
  `heapq`, `sorted`).
- A different selection family: radix select, histogram or threshold
  passes, a global sort of the whole input followed by slicing, or a
  single-program sequential selection; each replaces the hierarchical
  block tournament, and with it the passes and the work decomposition
  that define this operator.
- A per-block candidate width other than `K'` (a width below `k` breaks
  exactness; any other width changes the inter-level candidate traffic), or
  a fan-in below 2 (a level that does not shrink the candidate set).
- Returning more or fewer than `k` values, unsorted or ascending output, or
  an output that aliases `input`.
- Mutating `input`.

## Permitted PyTorch operations

- `torch.empty((nb, K'), dtype=torch.float32, device=...)` for the
  per-level candidate buffers (or one reused scratch).
- `Tensor.contiguous` as a no-op guard on the input; `Tensor.view` /
  `Tensor.reshape` to flatten a level's buffer; basic slicing
  (`buf[0, :k]`) to form the result; `Tensor.numel`, `.shape`, `.dtype`,
  `.device`, `.is_cuda`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`: no sorting,
selection, reduction, comparison or arithmetic on tensors through PyTorch.
