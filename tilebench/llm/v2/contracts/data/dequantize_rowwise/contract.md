# dequantize_rowwise: canonical algorithm contract

## Functional semantics
Row-wise int8 dequantisation with a per-row fp32 scale (bitsandbytes style).
Given `x` of shape (rows, cols) in int8 and `state_x` of shape (rows,) in
fp32:

    out[r, c] = fp16( fp32(x[r, c]) * state_x[r] * (1 / 127) )

i.e. `(state_x.unsqueeze(1) * x * (1.0 / 127.0)).to(torch.float16)`. The
scale is applied to every element of the row; the constant 1/127 is the
int8 quantisation step. There is no offset, no clipping and no per-column
scale.

## Inputs and outputs
- `x`: (rows, cols), int8, row-major contiguous (as provided; may be assumed).
- `state_x`: (rows,), fp32, contiguous.
- Output: exactly one tensor of shape (rows, cols) in fp16 (the output dtype
  is fixed to fp16 and does not follow the input dtype), freshly allocated
  inside run() on every call; no aliasing with any input.
- No input may be modified.
- Call form: `run(x, state_x, autotune=False, **kwargs)`. `autotune` and any
  `block_size` keyword are framework knobs: ignore them and never run a
  configuration search.

## Required logical stages
1. Element-wise dequantise: for each element, convert the int8 value to fp32,
   multiply by the fp32 row scale and by the fp32 constant 1/127.
2. Round the fp32 result to fp16 and store it at the same (r, c) position.

Stage 2 follows stage 1 per element. Both stages execute in one launch; there
is no reduction, no intermediate and no second pass. The row scale is a
broadcast operand: it may be loaded once per program and reused across every
element of that row handled by the program.

## Algorithm family and structure
Memory-bound streaming map with a per-row broadcast scalar: one read of the
int8 matrix, one read of the scale vector, one fp16 write. No reduction,
scan or sort of any kind.

## Precision and accumulation
- Convert x and the scale to fp32 before multiplying; perform both multiplies
  in fp32.
- The 1/127 factor is applied as a multiplication by the reciprocal held as
  an fp32 constant; dividing by 127 in fp32 is also acceptable (the
  difference is within the configured tolerance).
- Exactly one rounding, from fp32 to fp16, at the store. Do not form the
  product in int8/fp16 or round an intermediate.
- Tolerance: the operator config's verify section.

## Preprocessing and timing boundary
run() performs only shape reads, the fp16 output allocation and the launch.
No host-side casts of x or state_x, no copies, no transposes, no padding
or packing, no state across calls. Nothing may be precomputed outside
run(), and no cached derived operand (e.g. a pre-cast fp32 copy of x) is
allowed.

## Permitted implementation mappings
- Column chunk width per program, number of rows per program, launch
  geometry, vector width, number of concurrently resident programs.
- Whether the scale is loaded as a scalar or as a one-element tile and how
  it is broadcast.
- Masking of a partial last chunk versus zero-padded loads with clipped
  stores (padded lanes must never be written).
- Specialising the kernel on cols or keeping it a runtime argument.

## Forbidden substitutions
- Computing the product on the host with tensor arithmetic or any library
  dequantisation (torch.dequantize, quantized tensor constructors,
  torch.mul/torch.div on the inputs, broadcasting expressions on tensors).
- Changing the output dtype, or the rounding point (e.g. int8 * fp16 scale
  in fp16).
- Any pass that writes an intermediate buffer (e.g. a fp32 copy of x) to
  global memory.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty` for the fp16 output.
- Tensor metadata: `.shape`, `.stride()`, `.dtype`, `.device`, `.numel()`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden; in particular no `.to(...)`, `.float()`,
`.contiguous()` with data movement, `.view` or `.reshape` of the inputs is
needed or permitted.
