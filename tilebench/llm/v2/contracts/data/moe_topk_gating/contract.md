# moe_topk_gating: canonical algorithm contract

## Functional semantics

Given router logits of shape `(M, E)` (`M` tokens, `E` experts) and an
integer `k`, select for every row the `k` largest logits together with their
column indices, in descending value order, and turn the selected values into
gating weights with a softmax taken over only those `k` values:

```
vals[r, :], idx[r, :] = the k largest entries of logits[r, :], descending
weights[r, :] = softmax(float32(vals[r, :]))      # over the k values only
return (weights cast to logits.dtype, idx as int32)
```

This is the reference `torch.topk(logits, k, dim=-1, largest=True,
sorted=True)` followed by `softmax(vals.float()).to(logits.dtype)` and
`idx.to(torch.int32)`.

The entry point is called as `run(logits, M, E, k)` where
`logits.shape == (M, E)`. No keyword arguments are passed.

## Inputs and outputs

- `logits`: `(M, E)`, contiguous row-major, dtype one of fp16, bf16, fp32.
  Read-only: it must not be written and must not be copied on the host.
- Returned value: a 2-tuple `(weights, indices)` in this order.
  - `weights`: `(M, k)`, dtype equal to `logits.dtype`, freshly allocated in
    `run()`.
  - `indices`: `(M, k)`, dtype int32, freshly allocated in `run()`.
- Within a row the values are distinct in the benchmark's inputs; behaviour
  under exact ties is outside the verified input domain.
- `E` and `k` are small enough that a whole row fits in one program's
  working set; `E` and `k` need not be powers of two.

## Required logical stages

1. **Allocate** both outputs.
2. **Row load**: read the row of `E` logits once and convert it to float32.
   Positions beyond `E` in a padded working tile must hold `-inf` so they
   can never be selected.
3. **Selection**: `k` sequential rounds; each round finds the maximum of the
   remaining row and its column index, records both in the next output slot,
   and excludes that column from later rounds by overwriting it with `-inf`.
   Slot order is therefore descending value order.
4. **Softmax over the selected values**: in float32, subtract the maximum of
   the `k` values, exponentiate, divide by the sum of the `k` exponentials.
   Any padded slot beyond `k` must contribute exactly zero to the sum.
5. **Store**: write the `k` weights cast to `logits.dtype` and the `k`
   int32 indices for the row.

Stages 2 to 5 must be fused into one launch per row group: every row is
independent, and nothing is written to global memory between the stages.
Within a row, round `i` of stage 3 depends on round `i - 1`, and stage 4
depends on all rounds.

## Algorithm family and structure

Row-parallel selection by repeated maximum extraction with exclusion masking
(`k` rounds of full-row max and argmax), followed by a numerically stable
softmax over the `k` selected values. The reductions inside a round may use
any tree order. The selection must read the row from global memory once and
keep it on chip across the rounds; the softmax must be over the `k` selected
values only, never over all `E` experts.

## Precision and accumulation

- Selection comparisons are performed in float32 after an exact upcast of
  fp16/bf16 inputs, so the selected set and order equal those on the native
  dtype.
- Softmax arithmetic (max subtraction, exp, sum, division) is float32.
- Weights are cast to `logits.dtype` only when stored; indices are int32.
- Masking values: `-inf` for columns beyond `E`, for already-selected columns
  and for unfilled selection slots; `0` for unfilled index slots.

## Preprocessing and timing boundary

Everything happens inside `run()` and is timed: the two output allocations,
derivation of padded tile widths from `E` and `k`, and the fused kernel. No
cast, copy, sort or partial preprocessing of `logits` may happen on the host,
and nothing may be cached across calls.

## Permitted implementation mappings

- Rows per program, the padded tile width (any power of two at least `E`),
  the number of lanes and launch geometry are free.
- Whether the `k` rounds are unrolled at compile time or executed as a loop
  is free; `k` may be specialised as a compile-time constant.
- The tree order of max, argmax and sum reductions, and the exp
  implementation, are free.
- Recording the selected value and index into slot `i` by a lane-select, a
  scalar register or any equivalent mechanism is free.
- Masking by explicit masks or by the DSL's `-inf`-padded loads is free.

## Forbidden substitutions

- Any PyTorch selection or sorting on the host: `torch.topk`, `torch.sort`,
  `torch.argsort`, `torch.max`, `torch.argmax`, `torch.kthvalue`,
  `torch.nn.functional.softmax`, `torch.softmax`, `torch.exp`, or
  index-based gathers that assemble the outputs.
- Computing the softmax over all `E` logits and then gathering `k` entries
  (different semantics).
- Performing the selection in the native fp16/bf16 dtype, or the softmax in
  a dtype narrower than float32.
- Writing the row or intermediate selections to global scratch between
  stages.
- Mutating `logits`.

## Permitted PyTorch operations

- `torch.empty((M, k), dtype=logits.dtype, device=logits.device)` and
  `torch.empty((M, k), dtype=torch.int32, device=logits.device)` for the
  outputs.
- Reading `logits.shape`, `logits.dtype`, `logits.device`, and obtaining the
  current stream.
- Host-side integer arithmetic on `E` and `k` (next power of two).

Everything else in `torch` is forbidden inside `run()`.
