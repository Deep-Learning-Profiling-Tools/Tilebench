# 2d_max_pooling: canonical algorithm contract

## Functional semantics

2-D max pooling over a flat NCHW tensor. Interpreting `input` as
`(N, C, H, W)`:

```
out[n, c, oh, ow] = max over kh < kernel_size, kw < kernel_size of
    x[n, c, oh*stride + kh - padding, ow*stride + kw - padding]
```

taken over in-bounds positions only; positions outside `[0, H) x [0, W)`
behave as `-inf` and never win. `H_out = (H + 2*padding - kernel_size) //
stride + 1` and likewise `W_out` (floor, no ceil mode, dilation 1). This is
exactly `torch.nn.functional.max_pool2d(input.view(N, C, H, W),
kernel_size, stride=stride, padding=padding).reshape(-1)`.

The entry point is called as `run(input, N, C, H, W, kernel_size, stride,
padding)`; the seven ints are positional problem parameters, not tunables.
The entry point is called positionally with exactly the inputs listed below;
no keyword arguments are passed (declared keyword parameters, if any, must
have defaults and must not affect the computation).

## Inputs and outputs

- `input`: flat `(N*C*H*W,)`, contiguous, dtype fp16, bf16 or fp32; NCHW
  order (plane `n*C + c` occupies `H*W` consecutive elements). Read-only.
  Every configured case has `W == H`; supporting shapes outside the task's configured cases is not required.
- `kernel_size`, `stride`, `padding`: ints; the same value applies to both
  spatial axes; padding is symmetric.
- Output: flat `(N*C*H_out*W_out,)` in NCHW order, dtype of `input`,
  freshly allocated inside `run()` on every call, returned as a single
  tensor. It must not alias `input`. `input` must not be written.
- If `N*C*H_out*W_out <= 0` an empty tensor of the input dtype may be
  returned (never exercised by the benchmark).

## Required logical stages

1. **Allocate** the flat output (uninitialised allocation suffices).
2. **Direct windowed max**: one logical pass in which every output element
   takes the maximum over its `kernel_size*kernel_size` window, reading the
   input in place with `-inf` for out-of-range taps, and is written exactly
   once.

Stage 2 is the only device work. No pass may materialise a padded copy of
the input or an unfolded window matrix, and the output must not be produced
by a second pass over partial results, because each of these adds a
global-memory intermediate and an extra logical pass that the direct stencil
does not have. Spreading the output tiles over more than one launch, each
reading the input directly and writing final values, is a mapping choice.

## Algorithm family and structure

Direct stencil (sliding-window) reduction. Each output tile keeps a running
maximum that starts from `-inf` (the identity of the maximum), folds the
`kernel_size*kernel_size` taps in with an elementwise maximum, and is stored
with bounds masking wherever the tile overhangs the output; how many output
rows, columns or planes one program covers is a mapping choice. The tap
visiting order is free (the maximum is order-independent for non-NaN
values). Taps whose input coordinate is negative or `>= H`/`>= W` must
contribute `-inf` (masked loads with `-inf` fill, or an equivalent
bounds-padded gather); this covers both the padding ring and windows
overhanging the last row/column. NaN behaviour is unspecified; the benchmark
inputs contain no NaN.

## Precision and accumulation

No arithmetic is performed. The running maximum may be kept in the input
dtype (the maximum is exact in any dtype) or in a wider type; the output is
stored in the input dtype. `-inf` is the only fill/identity value and is
representable in all three dtypes.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

`run()` must not pad, unfold, cast or copy the input and must not cache
anything across calls. It may compute `H_out`, `W_out` and the output size,
allocate the output, take zero-copy views of the flat tensors (for example
as `(N*C, H, W)` and `(N*C, H_out, W_out)`), and launch. The metric formulas
count one read of the input and one write of the output, plus one comparison
per tap per output; the cache re-reads of overlapping windows are expected
and not counted.

## Permitted implementation mappings

- Output tile shape (1-D runs of a row or 2-D row-by-column tiles), planes
  per program, grid shape, raster order, vector width, pipelining.
- Element-wise gathers with bounds padding, or masked block loads, for the
  taps; static unrolling of the tap loops; making `kernel_size`, `stride`
  and `padding` compile-time constants of the kernel.
- Tail handling of partial edge tiles by explicit masks or by the DSL's
  bounds-clipped stores.
- The number of launches over which the output tiles are spread, provided
  each launch reads the input directly and writes final output values.

## Forbidden substitutions

- `torch.nn.functional.max_pool2d` / `max_pool1d` / `max_pool3d`,
  `torch.nn.MaxPool2d`, `torch.amax` / `torch.max` over an unfolded window
  tensor, or any other library pooling.
- `unfold` / im2col-style window materialisation, `torch.nn.functional.pad`
  or any padded copy of the input (each materialises an intermediate in
  global memory that the direct stencil does not have).
- A multi-pass scheme that produces the output from partial results kept in
  global memory (for example a row-wise max followed by a column-wise max
  written through scratch), because it adds a global round trip and a second
  logical pass over the data.
- Any dtype conversion of the stored result; mutating `input`; returning a
  view of `input`.

## Permitted PyTorch operations

- `torch.empty(N*C*H_out*W_out, dtype=input.dtype, device=input.device)`
  for the output (and `torch.empty(0, ...)` for the degenerate empty case).
- `.view(...)` / `.reshape(...)` on the contiguous flat input and output as
  metadata-only reshapes.
- Reading dtype, shape and device metadata, and obtaining the current
  stream.

Everything else in `torch` is forbidden inside `run()`.
