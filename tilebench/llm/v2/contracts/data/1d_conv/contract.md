# 1d_conv: canonical algorithm contract

## Functional semantics

Grouped 1-D cross-correlation without bias, identical to
`torch.nn.functional.conv1d(input, weight, bias=None, stride=stride,
padding=padding, groups=groups)`. With `Cin_g = in_channels // groups`,
`Cout_g = out_channels // groups` and `g = oc // Cout_g`:

```
out[b, oc, ol] = sum over ic_local < Cin_g, kl < kernel_size of
    input[b, g*Cin_g + ic_local, ol*stride + kl - padding] * weight[oc, ic_local, kl]
```

Input positions outside `[0, L)` contribute exactly zero (implicit symmetric
zero padding). `out_L = (L + 2*padding - kernel_size) // stride + 1`. The
kernel is not flipped. There is no bias.

The entry point is called as `run(input, weight, stride, padding, groups,
**kwargs)` with the three scalars positional; it must accept and ignore the
framework keywords `block_size` and `autotune` and any extra keywords.

## Inputs and outputs

- `input`: `(batch, in_channels, L)`, contiguous, dtype fp16 or fp32. Read-only.
- `weight`: `(out_channels, in_channels // groups, kernel_size)`, contiguous,
  same dtype as `input`. Read-only.
- `stride`, `padding`, `groups`: Python ints. `in_channels` is a multiple of
  `groups`; `out_channels` is a multiple of `groups`.
- Output: `(batch, out_channels, out_L)`, contiguous, dtype of `input`,
  freshly allocated inside `run()` on every call, returned as a single
  tensor. It must not alias any input. Neither input may be written.

## Required logical stages

1. **Allocate** the output (uninitialised allocation suffices: every element
   is written exactly once by stage 2). Host shape arithmetic (`out_L`,
   channels per group) is metadata only.
2. **Implicit-GEMM convolution**: one device pass that, for each group,
   treats the output as a `(batch*out_L) x Cout_g` matrix, the reduction
   axis as the flattened `(ic_local, kl)` pair of length
   `Cin_g * kernel_size`, gathers the input patch and weight tiles on the fly
   with zero fill for out-of-range positions, multiplies them with a
   matrix-multiply primitive into a local fp32 accumulator, and stores the
   cast result.

Stage 2 is the only device work and is a single launch. It must not be
split into an im2col/unfold pass plus a GEMM, a padded-copy pass plus a
convolution, or a split-K partial pass plus a reduce pass.

## Algorithm family and structure

Implicit GEMM (im2col-free) grouped convolution. Each output element is
produced by exactly one local accumulator that sums the full reduction axis
sequentially in chunks; within a chunk the product is a tensor-core (or
equivalent) matrix multiply. The order in which the flattened reduction axis
is enumerated (channel-major or tap-major) is free and only changes the
summation order. No split-K, no atomics, no partial-sum scratch. Groups are
independent GEMMs and may be mapped to a grid axis or folded into the row
index. Out-of-range input positions (padding region), out-of-range reduction
lanes and out-of-range output-channel lanes must contribute exactly zero:
zero-filled masked loads, clamped addresses plus masking, or an equivalent.
Out-of-range output lanes must not be stored (mask, clip, or redirect to a
discarded index).

## Precision and accumulation

- Accumulation is fp32 for every output element over the whole reduction,
  for every input dtype.
- fp16 inputs: native fp16 products, fp32 accumulation.
- fp32 inputs: the expected precision class is a tensor-core product with
  operands rounded to TF32 and fp32 accumulation; the verification tolerance
  is sized for this. A full-fp32 product is permitted but not required.
  Casting fp32 operands to fp16 or bf16 is forbidden.
- The fp32 accumulator is cast to the input dtype at the store.
- Zero is the only fill value; no `-inf`/NaN sentinels are involved.

## Preprocessing and timing boundary

Everything `run()` does is timed. It must not cast, transpose, pack, pad or
copy `input` or `weight`, must not materialise an im2col matrix, and must not
cache anything derived from input or weight contents across calls. It may
compute scalar metadata, allocate the output, take zero-copy views of the
contiguous tensors, and launch. The metric formulas count one read of
`input`, one read of `weight` and one write of the output; the redundant
re-reads of overlapping input windows that an implicit GEMM performs through
the cache are expected and not counted.

## Permitted implementation mappings

- Tile extents along the three GEMM axes, reduction chunk width, grid shape,
  raster order, pipelining depth, compile-time versus runtime trip counts.
- Element-wise gathers with computed addresses, structured block loads, or
  staged loads for the input-patch and weight tiles.
- How out-of-range addresses are made safe (clamp plus mask, zero-fill loads,
  index redirection) and how out-of-range output lanes are suppressed.
- Flattening contiguous tensors to 1-D views and passing explicit strides.

## Forbidden substitutions

- Any library convolution or GEMM (`torch.nn.functional.conv1d`/`conv2d`/
  `conv3d`, `torch.conv1d`, `torch.matmul`, `torch.bmm`, `torch.mm`,
  `torch.einsum`, cuDNN/cuBLAS).
- Materialising im2col/unfold, a Toeplitz matrix, or a padded copy of the
  input.
- Split-K with a second reduce pass, or atomic accumulation into the output.
- Accumulating in less than fp32, or casting fp32 operands below TF32
  precision.
- Writing to `input` or `weight`, or returning a view of either.

## Permitted PyTorch operations

- `torch.empty((batch, out_channels, out_L), dtype=input.dtype,
  device=input.device)` for the output.
- `.view(-1)` / `.reshape(-1)` on the contiguous inputs and output as
  metadata-only flattening; `.stride()`, `.shape`, `.dtype`, `.device`,
  `.is_contiguous()` for metadata; obtaining the current stream.

Everything else in `torch` is forbidden inside `run()`.
