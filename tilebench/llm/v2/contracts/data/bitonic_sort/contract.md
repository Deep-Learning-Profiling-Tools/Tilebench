# bitonic_sort: canonical algorithm contract

## Functional semantics

Ascending sort of a 1-D array of `N` values: element `i` of the result is
the `i`-th smallest value of `data` (values only, no indices). The reference
realises this with an explicit bitonic sorting network on a `+inf`-padded
power-of-two buffer, and the result equals the ascending values of a sort
of `data`. For `N <= 1` a copy of `data` is returned (never exercised).

The entry point is called as `run(data, N, **kwargs)`; `N` is positional and
equals `data.shape[0]`. It must accept and ignore `block_size` and
`autotune`.

## Inputs and outputs

- `data`: `(N,)`, contiguous, dtype fp16 or fp32. Read-only: it must never
  be written, and the result must not share storage with it.
- Output: `(N,)`, ascending, dtype of `data`. It is a zero-copy leading
  slice of the working buffer allocated inside `run()` (length
  `M = next power of two >= N`); the caller receives a view whose backing
  storage is `M` elements long. Returned as a single tensor.

## Required logical stages

1. **Allocate** the working buffer `work` of `M` elements in the dtype of
   `data` (uninitialised allocation suffices).
2. **Pad**: one device pass that writes `data[0:N]` into `work[0:N]` and
   `+inf` into `work[N:M]`.
3. **Compare-exchange network**: for `k = 2, 4, ..., M` in ascending order
   and, for each `k`, `j = k/2, k/4, ..., 1` in descending order, stage
   `(k, j)` performs, for every index `i < M` whose partner `p = i XOR j`
   satisfies `p > i`: `ascending = (i AND k) == 0`; swap `work[i]` and
   `work[p]` when `ascending and work[i] > work[p]` or
   `not ascending and work[i] < work[p]`. Comparisons are strict: equal
   values are never exchanged. Each pair is exchanged exactly once per
   stage, by the lane owning the lower index. There are `L*(L+1)/2` stages
   for `L = log2(M)`.
4. **Return** `work[:N]` as a zero-copy view.

Dependencies and fusion: stage `(k, j)` for index `i` depends on the
preceding stage's results for both `i` and `i XOR j`, so stages are
separated by a global synchronisation (a launch boundary) unless every
compare-exchange of a group of consecutive stages is confined to data owned
by one program (an aligned power-of-two slice longer than every partner
distance `j` in the group); such a group may be executed by that program
locally. The pad pass may be fused into the first such group. At every
global synchronisation point the entire array lives in `work`; the network
is executed in place and no second buffer is permitted.

## Algorithm family and structure

Bitonic sorting network over a `+inf`-padded power-of-two array: the
standard `k`-ascending / `j`-descending schedule, partner by XOR, direction
by `(i AND k) == 0`, strict comparisons. Padding values sink to `work[N:M]`
and are dropped by the final slice. Stability is unobservable because only
values are sorted. Any other sorting algorithm (radix, merge, odd-even,
sample sort), any other comparator schedule, and non-strict comparisons are
not this algorithm.

## Precision and accumulation

No arithmetic: values are compared and moved in the dtype of `data` with no
conversion. `+inf` is the only fill value and is representable in both
dtypes. Loads for lanes that are inactive in a stage may use any fill value
because those lanes are never written.

## Preprocessing and timing boundary

Everything `run()` does is timed: the allocation of `work`, the pad pass,
every stage of the network and the final zero-copy slice. Nothing may be
cached across calls and nothing may be computed on the host beyond `M` and
the `(k, j)` schedule. The metric formulas count one read and one write of
`N` elements (the compulsory traffic of a sort); the pad pass and the
multiple passes over `M` elements that the network performs are expected
and not counted.

## Permitted implementation mappings

- Elements per program, grid shape, vector width, pipelining.
- Whether `k` and `j` are runtime arguments or compile-time specialisations
  of the stage kernel.
- Block loads plus gathers for partner values, or two gathers; how inactive
  lanes are suppressed (masks, or redirection of their store index to a
  discarded location).
- Fusing consecutive stages inside a program under the dependency rule
  above, and fusing the pad pass into the first stage group.

## Forbidden substitutions

- `torch.sort`, `torch.argsort`, `torch.msort`, `torch.topk`,
  `torch.kthvalue`, Python `sorted`, or any library sort.
- A different sorting algorithm or comparator schedule; non-strict
  comparisons; a direction rule other than `(i AND k) == 0`.
- Performing the pad with library operations (`torch.full` plus slice
  assignment, `torch.cat`, `torch.nn.functional.pad`, `.clone()`) instead
  of a kernel pass; a second working buffer or ping-pong buffers.
- Sorting a copy of `data` in place and returning `data`; writing `data`;
  returning a tensor that shares storage with `data`.
- Any dtype conversion.

## Permitted PyTorch operations

- `torch.empty((M,), dtype=data.dtype, device=data.device)` for the
  working buffer.
- `work[:N]` (leading slice) and `.contiguous()` on that slice, which is a
  zero-copy view because the slice starts at offset zero with unit stride.
- `data.clone()` only on the `N <= 1` early-exit path.
- Reading dtype, shape and device metadata, and obtaining the current
  stream.

Everything else in `torch` is forbidden inside `run()`.
