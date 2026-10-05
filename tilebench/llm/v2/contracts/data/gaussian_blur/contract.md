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
- Call form: `run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, block_size=..., autotune=False, **kwargs)`.
  The keyword knobs are framework knobs; ignore them and never run a
  configuration search.

## Required logical stages
1. For each output element, accumulate the kernel_rows * kernel_cols tap
   products from the zero-padded neighbourhood into an fp32 accumulator.
2. Round the accumulator to the input dtype and store it at the same flat
   position.

Both stages execute in ONE launch; there is no intermediate buffer, no
separate padding pass and no second pass over the image. Each output element
is produced by exactly one program.

## Algorithm family and structure
Direct 2-D stencil / small-kernel cross-correlation: per logical output tile,
the shifted input neighbourhood is read for each tap (or staged once on-chip
with a halo) and multiplied by the tap weight. The sum over taps is a
sequential fp32 accumulation per output element; the tap visiting order is
free. Out-of-image taps contribute exactly 0 (zero padding), which also
covers tile overhang at the image border. No reduction across programs, no
scan, no sort.

## Precision and accumulation
- The accumulator over taps is fp32 for both input dtypes.
- The accumulated fp32 value is rounded once, to the input dtype, at the
  store.
- Whether the per-tap product itself must be formed in fp32 for fp16 inputs
  (converting pixel and weight before multiplying) or may be formed in the
  input dtype before being added to the fp32 accumulator is an open review
  item; both are currently within the configured tolerance.
- Tolerance: the operator config's verify section.

## Preprocessing and timing boundary
run() performs only: the element-count guard, the output allocation,
metadata-only 2-D views of the flat input/output, and the launch. No
host-side padding of the image, no im2col/unfold buffer, no casts, copies
or re-layouts of the image or the kernel, no state across calls, nothing
precomputed outside run(). The kernel weights are read inside the launch
(as scalars per tap or as a small tile), never pre-expanded on the host.

## Permitted implementation mappings
- Output tile shape, launch geometry, number of resident programs, vector
  width along columns.
- Per-tap masked/padded loads or gathers of the shifted tile versus staging
  a haloed input tile on-chip once and reading taps from it.
- Statically unrolled tap loops (kernel size as a compile-time constant)
  versus runtime loops.
- Loading each tap weight when needed or all weights once per program.

## Forbidden substitutions
- Any library convolution or correlation (torch.nn.functional.conv2d,
  conv1d, torch.conv2d, scipy/other filters) or FFT-based filtering.
- im2col / unfold followed by a matrix product; separable (row/column)
  decomposition of the kernel (the kernel is not separable in general); a
  kernel flip (true convolution).
- Host-side zero-padding of the image or any intermediate image buffer.
- Non-zero padding modes (reflect, replicate, clamp).
- Accumulating in fp16, or rounding the accumulator between taps.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the output.
- `Tensor.view` / `.reshape` of the flat contiguous input and output to
  2-D (metadata only).
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.

## Open review items
- Per-tap product precision for reduced-precision inputs: must pixel and
  weight be converted to fp32 before each multiply, or may the product be
  formed in the input dtype before fp32 accumulation?
