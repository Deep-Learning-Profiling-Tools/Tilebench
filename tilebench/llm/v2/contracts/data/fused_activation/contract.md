# fused_activation: canonical algorithm contract

## Functional semantics
Fused gated activation. Given three same-shape vectors `x`, `gate` and
`bias` of length n:

    z[i]   = x[i] * gate[i] + bias[i]
    out[i] = z[i] * sigmoid(z[i]) = z[i] / (1 + exp(-z[i]))

i.e. `torch.nn.functional.silu(x * gate + bias)`. `bias` is a full
per-element tensor (same shape as x), not a broadcast scalar or vector.

## Inputs and outputs
- `x`, `gate`, `bias`: (n,), fp32 (the only configured dtype), contiguous,
  all of identical shape (an implementation may raise on a mismatch).
- Output: exactly one tensor of x's shape in fp32, freshly allocated inside
  run() on every call; no aliasing with any input.
- No input may be modified.
- Call form: `run(x, gate, bias)`, positional; no keyword arguments are
  passed. Never run a configuration search.

## Required logical stages
1. Element-wise fused multiply-add and SiLU: for each element read x, gate
   and bias, compute z = x*gate + bias and out = z * sigmoid(z) in fp32, and
   store.

A single launch; no reduction, no intermediate, no second pass. The three
input reads for an element and the write of its result belong to the same
program.

## Algorithm family and structure
Memory-bound streaming map with three input streams and one output stream.
No reduction, scan or sort. Elements are independent, so any partitioning
of the index range across programs is acceptable.

## Precision and accumulation
- fp32 for the multiply-add, the exponential/sigmoid and the final product;
  inputs are read as fp32 (or converted to fp32 immediately after loading).
- SiLU may be written as z * sigmoid(z) with a sigmoid intrinsic, as
  z / (1 + exp(-z)), or with an equivalent base-2 exponential; large
  negative z must yield 0 (not NaN), which all of these forms do.
- The result is stored in fp32 (no intermediate rounding).
- Tolerance: the operator config's verify section.

## Preprocessing and timing boundary
run() performs only: the shape check, the output allocation and the launch.
`.contiguous()` on the inputs is permitted only as a no-op (they are
contiguous). No casts, copies, packing or fusion of the three inputs into
one buffer on the host; no state across calls; nothing precomputed outside
run().

## Permitted implementation mappings
- Elements per program, launch geometry, vector width, number of resident
  programs.
- Masked tail handling versus zero-padded loads with clipped stores (padded
  lanes evaluate to silu(0) = 0 and must never be written).
- Order and interleaving of the three loads; algebraic form of SiLU.

## Forbidden substitutions
- torch.nn.functional.silu / torch.nn.SiLU, torch.sigmoid, torch.special.expit
  or any host-side tensor arithmetic (x * gate + bias, torch.addcmul,
  torch.exp on tensors) for any part of the computation.
- Splitting the FMA and the activation into separate launches or writing z
  to global memory.
- Reduced-precision evaluation (fp16/bf16) of z or the activation.
- Any buffer other than the output.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the fp32 output.
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`.
- `Tensor.contiguous()` only as a no-op.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
