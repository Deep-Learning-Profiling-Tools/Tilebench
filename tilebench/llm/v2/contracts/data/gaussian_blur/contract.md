# gaussian_blur: canonical algorithm contract

## Functional semantics
Single-channel 2-D "same"-size filtering of a flat image with a small
kernel, using cross-correlation (no kernel flip) and zero padding. Given a
flat row-major image `input` of input_rows * input_cols elements and a flat
row-major kernel `kernel` of kernel_rows * kernel_cols elements:

    out[r, c] = sum_{i < kernel_rows} sum_{j < kernel_cols}
                x[r + i - kernel_rows//2, c + j - kernel_cols//2] * w[i, j]

with x[...] = 0 whenever the index lies outside the image. This equals
`torch.nn.functional.conv2d(x.view(1,1,R,C), w.view(1,1,kr,kc),
padding=(kr//2, kc//2))` cropped to the first R rows and C columns and
flattened. In the benchmark the kernel is 7x7, non-negative and normalised
to sum 1, and images are square; the implementation must nevertheless use
the kernel_rows / kernel_cols / input_rows / input_cols it is given.

## Inputs and outputs
- `input`: (input_rows * input_cols,), fp16 or fp32, contiguous, row-major
  image.
- `kernel`: (kernel_rows * kernel_cols,), same dtype as input, contiguous,
  row-major.
- `input_rows`, `input_cols`, `kernel_rows`, `kernel_cols`: Python ints.
- Output: exactly one flat tensor of input_rows * input_cols elements in
  input's dtype, freshly allocated inside run() on every call (an empty
  tensor if the image has no elements); no aliasing with any input.
- No input may be modified.
- Call form: `run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols)`,
  positional; no keyword arguments are passed. Never run a configuration search.

## Required logical stages
1. For each output element, accumulate the kernel_rows * kernel_cols tap
   products from the zero-padded neighbourhood into an fp32 accumulator,
   each product formed in fp32 from the fp32-converted pixel and weight.
2. Round the accumulator to the input dtype and store it at the same flat
   position.

Both stages belong to one logical pass: there is no intermediate buffer, no
separate padding pass and no second pass over the image, because each of
these would add a global-memory round trip that the direct stencil does not
have. The whole tap sum of an output element is accumulated within one
program (no partial sums combined across programs), because the canonical
algorithm keeps it in a single on-chip fp32 accumulator. Spreading the
output tiles over more than one launch, each producing final values
directly from the image, is a mapping choice.

## Algorithm family and structure
Direct 2-D stencil / small-kernel cross-correlation: per logical output tile,
the shifted input neighbourhood is read for each tap (or staged once on-chip
with a halo) and multiplied by the tap weight. The sum over taps is a
sequential fp32 accumulation per output element; the tap visiting order is
free. Out-of-image taps contribute exactly 0 (zero padding), which also
covers tile overhang at the image border. No reduction across programs, no
scan, no sort.

## Precision and accumulation
- For both input dtypes (fp16 and fp32), the pixel and the tap weight are
  converted to fp32 before each multiply, and the products are accumulated
  in fp32: no per-tap product is formed in a dtype narrower than fp32.
- The accumulated fp32 value is rounded once, to the input dtype, at the
  store.
- Tolerance: the operator config's verify section.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Besides launching the stencil, run() performs only host-side work: the
element-count guard, the output allocation and metadata-only 2-D views of
the flat input/output. No host-side padding of the image, no im2col/unfold
buffer, no casts, copies or re-layouts of the image or the kernel, no state
across calls, nothing precomputed outside run(). The kernel weights are read
on the device by the stencil computation (as scalars per tap or as a small
tile), never pre-expanded on the host.

## Permitted implementation mappings
- Output tile shape, launch geometry, number of resident programs, vector
  width along columns.
- Per-tap masked/padded loads or gathers of the shifted tile versus staging
  a haloed input tile on-chip once and reading taps from it.
- Statically unrolled tap loops (kernel size as a compile-time constant)
  versus runtime loops.
- Loading each tap weight when needed or all weights once per program.
- The number of launches over which the output tiles are spread, as long as
  each output element is produced in full by one program directly from the
  image.

## Forbidden substitutions
- Any library convolution or correlation (torch.nn.functional.conv2d,
  conv1d, torch.conv2d, scipy/other filters) or FFT-based filtering (a
  different algorithm family).
- im2col / unfold followed by a matrix product (materialises the patch
  matrix in global memory); separable (row/column)
  decomposition of the kernel (the kernel is not separable in general); a
  kernel flip (true convolution).
- Host-side zero-padding of the image or any intermediate image buffer
  (a global-memory round trip that the direct stencil does not have).
- Non-zero padding modes (reflect, replicate, clamp).
- Forming a per-tap product in a dtype narrower than fp32 (multiplying the
  fp16 pixel and weight before converting them), accumulating in fp16, or
  rounding the accumulator between taps.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the output.
- `Tensor.view` / `.reshape` of the flat contiguous input and output to
  2-D (metadata only).
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
