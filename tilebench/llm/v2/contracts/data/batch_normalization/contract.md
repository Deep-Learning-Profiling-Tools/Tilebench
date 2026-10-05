# batch_normalization: canonical algorithm contract

## Functional semantics

Training-mode batch normalisation of a 2-D `(N, C)` matrix over the batch
axis with a per-channel affine transform, exactly
`torch.nn.functional.batch_norm(input, running_mean=None,
running_var=None, weight=gamma.float(), bias=beta.float(), training=True,
momentum=0.0, eps=eps)`. For every channel `c`:

```
mean_c = (1/N) * sum over n of x[n, c]
var_c  = (1/N) * sum over n of (x[n, c] - mean_c)^2        (biased)
y[n, c] = (x[n, c] - mean_c) * rsqrt(var_c + eps) * gamma_c + beta_c
```

No running statistics are produced or consumed. `N >= 2` in every case
(the reference's `N == 1` special case is out of scope).

The entry point is called as `run(input, gamma, beta, N, C, eps,
**kwargs)`; `N`, `C` are ints and `eps` a float, all positional. It must
accept and ignore `block_size` and `autotune`.

## Inputs and outputs

- `input`: `(N, C)`, contiguous row-major, dtype fp16, bf16 or fp32.
  Read-only.
- `gamma`, `beta`: `(C,)`, contiguous, same dtype as `input`. Read-only.
- Output: `(N, C)`, contiguous, dtype of `input`, freshly allocated inside
  `run()` on every call, returned as a single tensor. It must not alias any
  input.

## Required logical stages

1. **Allocate** the output and the declared fp32 scratch (uninitialised
   allocation suffices for all of them).
2. **Partial statistics** (first read of `input`): each program owns a
   contiguous block of rows and reduces it, per channel, to a partial sum
   and a partial sum of squares (or an equivalent single-pass moment pair)
   in fp32, written to the partial-statistics scratch.
3. **Per-channel combine**: for each channel, the partials are combined in
   fp32 into `mean` and `inv_std = rsqrt(max(var, 0) + eps)`, where
   `var = sumsq/N - mean^2` (or the equivalent merge of single-pass
   moments). Padded lanes must contribute zero.
4. **Apply** (second read of `input`): `y = x * scale + shift` with
   `scale = inv_std * gamma` and `shift = beta - mean * scale` (or the
   unfolded `(x - mean) * inv_std * gamma + beta`), computed in fp32 and
   stored in the input dtype.

Dependencies and fusion: stage 3 needs every partial of stage 2 (a global
synchronisation); stage 4 needs stage 3. Stage 3 may be its own launch or
may be folded into the prologue of stage 4 (each program of stage 4
re-combines the partials for all channels). Stages 2 and 4 must be
separate launches (or separated by an equivalent grid-wide synchronisation).
Stages 2 and 3 must not be fused by atomically accumulating channel totals:
the reduction must be the deterministic two-level form (per-block partials,
then a per-channel combine in a fixed order). `input` is read exactly twice
(once in stage 2, once in stage 4); no third pass is permitted.

## Algorithm family and structure

Two-pass (statistics, then apply) batch normalisation with a two-level
per-channel reduction. Level 1: within a row block, rows are folded into
per-channel fp32 vectors (tree or sequential order over sub-tiles). Level 2:
per channel, the block partials are reduced in a fixed order. The variance
is formed from single-pass moments (sum and sum of squares, or a Welford /
Chan merge of per-block (count, mean, M2) triples), clamped at zero before
the `rsqrt`; a centred second pass over `input` is not permitted. The apply
stage is elementwise.

## Precision and accumulation

- Partial sums, sums of squares, mean, variance, `inv_std`, `scale`, `shift`
  and the normalised value are all fp32 for every input dtype; `x`, `gamma`
  and `beta` are upcast to fp32 inside the kernels.
- `var` is clamped to be non-negative before adding `eps` and taking the
  reciprocal square root.
- The output is the fp32 result cast to the input dtype at the store.
- Rows beyond `N` and channels beyond `C` in padded tiles contribute zero
  to the sums and are never stored.

## Preprocessing and timing boundary

Everything `run()` does is timed, including the scratch allocations. It
must not cast `gamma`/`beta`/`input` on the host (upcasts happen in-kernel),
must not copy or transpose anything, and must not cache statistics or
scratch across calls. The metric formulas count two reads of `input`, one
write of the output and one read each of `gamma` and `beta`; the small fp32
scratch traffic (partials written once and read once, plus the two `(C,)`
vectors) is expected and not counted.

## Permitted implementation mappings

- Rows per partial block, sub-tile height, programs per launch, vector
  width, pipelining, power-of-two padding of the channel axis for tile
  loads.
- Whether stage 3 is a separate launch or folded into stage 4's prologue.
- Tree versus sequential order inside level-1 and level-2 reductions.
- Sum/sum-of-squares versus Welford-style single-pass moments.
- Masked loads with zero fill or bounds-padded block loads with zero
  padding; masked or clipped stores.

## Forbidden substitutions

- `torch.nn.functional.batch_norm`, `torch.nn.BatchNorm1d`/`2d`,
  `layer_norm`, `instance_norm`, `torch.var_mean`, `torch.var`,
  `torch.std`, `torch.mean`, `torch.sum` or any tensor-method reduction
  computing the statistics on the host.
- A centred (two-pass) variance that reads `input` a third time.
- Atomic accumulation of channel totals.
- Computing statistics or the normalisation in less than fp32; host-side
  `.float()` copies of the inputs.
- Mutating any input; returning a view of any input.

## Permitted PyTorch operations

- `torch.empty_like(input)` for the output.
- `torch.empty(...)` for the fp32 scratch: the partial sums and partial
  sums of squares of shape `(num_row_blocks, C)` (or one combined buffer)
  and the `mean` and `inv_std` vectors of shape `(C,)`.
- `.view(...)` / `.reshape(...)` on contiguous tensors and
  `input.contiguous()` only as a no-op guard, all metadata-only.
- Reading dtype, shape and device metadata, and obtaining the current
  stream.

Everything else in `torch` is forbidden inside `run()`.
