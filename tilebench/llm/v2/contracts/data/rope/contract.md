# rope: canonical algorithm contract

## Functional semantics

Half-split ("rotate-half", non-interleaved) rotary position embedding of a
query tensor with position-indexed coefficient tables. For `q` of shape
`(B, S, H, D)` and tables `cos`, `sin` of shape `(S, D/2)`, with
`q1 = q[..., :D/2]`, `q2 = q[..., D/2:]` and `c = cos[s, :]`,
`s_ = sin[s, :]` broadcast over batch and heads:

    out[b, s, h, :D/2] = q1 * c  - q2 * s_
    out[b, s, h, D/2:] = q2 * c  + q1 * s_

The reference unsqueezes the tables to `(1, S, 1, D/2)`, forms the two
products above and concatenates them along the last dimension, all in the
input dtype. The tables are inputs with arbitrary values: they do not
necessarily satisfy `cos^2 + sin^2 = 1` and must not be regenerated from
positions or angles. Element `i` of a head pairs with element `i + D/2` of
the same head; adjacent-pair (interleaved) rotation is a different operator.

## Inputs and outputs

The entry point is called as `run(q, cos, sin, block_size=None,
autotune=False)`: the three tensors positional, keywords may be ignored.

- `q`: `(B, S, H, D)`, contiguous row-major, dtype fp16 or fp32, `D` even.
  Read-only.
- `cos`, `sin`: `(S, D/2)`, contiguous, same dtype as `q`. Indexed by the
  sequence position only: when rows are flattened as `b*S + s`, the table
  row is `row % S`. Read-only.
- Returned: one new tensor of shape `(B, S, H, D)` and dtype of `q`. It must
  not be `q`, a view of `q`, or share storage with any input. A buffer
  allocated inside `run()` on every call, either an empty output written
  out-of-place or a copy of `q` that is then rotated in place, satisfies this.

## Required logical stages

1. **Table fetch**: for each `(b, s)` row, the `cos` and `sin` rows of
   position `s`.
2. **Pairwise rotation**: for every head and every `i < D/2`, read the pair
   `(q[i], q[i + D/2])` and write the two rotated outputs to the same
   positions of the output.

This is one logical stage of elementwise work and is expected as a single
launch. The copy of `q` that an in-place scheme needs is part of `run()`,
is timed, and may be fused away by writing out-of-place.

## Algorithm family and structure

Purely elementwise 2x2 rotation per `(b, s, h, i)` pair with shared
per-position coefficients; no reduction, scan or sort. Each output element
depends on exactly two input elements of the same head and one `(cos, sin)`
pair. Programs may cover any grouping of rows, heads and pairs; out-of-range
heads or pairs must be masked or zero-padded and never stored. Each output
element is written exactly once.

## Precision and accumulation

- The reference evaluates the two multiplies and the add/subtract in the
  input dtype, and so do the canonical implementations: no upcast is
  required. Evaluating in fp32 with one cast at the store is also accepted.
- The verifier tolerance for this operator is `atol 5e-3, rtol 5e-3`
  (config `verify`), which covers both choices for fp16.
- No accumulation; no other rounding steps.

## Preprocessing and timing boundary

All work is inside `run()` and timed: any clone or copy of `q`, the views
used to address halves (for example `(B*S, H, 2, D/2)` or `(B*S, H*D)`), the
output allocation and the launch. The tables are consumed as given; no table
transformation, no state cached across calls, nothing precomputed outside
`run()`.

## Permitted implementation mappings

- Out-of-place (read `q`, write the fresh output) or clone-then-in-place.
- Heads per program, rows per program, grid shape, pipelining, vector width.
- Handling `D/2` as one tile or as chunks with masking; whether `S`, `H`, `D`
  are compile-time constants or runtime arguments; explicit strides versus
  the contiguous layout.
- Computing in the input dtype or in fp32.

## Forbidden substitutions

- Recomputing or regenerating `cos`/`sin` from positions, frequencies or
  angles (`torch.cos`, `torch.sin`, `torch.polar`, `view_as_complex` and
  similar complex-number formulations).
- Performing the rotation with PyTorch tensor arithmetic on the host
  (slicing `q1`/`q2`, `*`, `-`, `+`, `torch.cat`, `unsqueeze` broadcasting)
  or calling a library rotary-embedding routine.
- Interleaved (even/odd) pairing instead of half-split pairing.
- Indexing the tables by anything other than the sequence position (for
  example by the flattened row index without the modulo `S`).
- Modifying `q`, `cos` or `sin`; returning `q` or a view of `q`.

## Permitted PyTorch operations

- `torch.empty_like(q)` / `torch.empty(...)` for the output, or
  `q.clone()` (optionally `.contiguous()`) as the output buffer of an
  in-place scheme.
- `Tensor.view` / `Tensor.reshape` / `Tensor.contiguous` for layout views
  of `q`, `cos`, `sin` and the output; `Tensor.stride`, `.shape`, `.dtype`,
  `.device`.
- Obtaining the current CUDA stream for launching.

Everything else in `torch` is forbidden inside `run()`: no arithmetic,
slicing-based computation, concatenation or conversion of tensors through
PyTorch.
