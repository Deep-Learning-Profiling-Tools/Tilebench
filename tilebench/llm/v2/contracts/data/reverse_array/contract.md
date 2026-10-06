# reverse_array: canonical algorithm contract

## Functional semantics

Given a 1-D tensor `input` of length `N`, produce a new contiguous tensor
`out` of the same length and dtype with `out[i] == input[N - 1 - i]` for
every `i`. This is the reference `input.flip(0).contiguous()`: a
materialised reversal, not a negative-stride view.

The entry point is called as `run(input, N)` with `N == input.numel()`. No
keyword arguments are passed.

## Inputs and outputs

- `input`: `(N,)`, contiguous, dtype one of fp16, bf16, fp32, int8.
  Read-only: it must not be written and the result must not alias it.
- `out` (returned): `(N,)`, same dtype as `input`, freshly allocated inside
  `run()` on every call, contiguous. Returned as a single tensor.
- Tail handling is required wherever the task's fixed `N` is not a
  multiple of the chosen tile: no element outside `[0, N)` may be read or written
  (note that the mirrored index of an out-of-range output position is
  negative).

## Required logical stages

1. **Allocate** the output (uninitialised allocation is sufficient).
2. **Mirrored copy**: one logical traversal that reads every element of
   `input` once and writes it once to its mirrored position in `out`.

Stage 2 is a single logical stage with no inter-element dependencies, and no
intermediate tensor is materialised in global memory. No additional pass over
the data is permitted, because it adds a full extra read and write of the
array and so changes the algorithm rather than its mapping. These are
logical traversal counts, not a guarantee about physical DRAM transactions,
which caches and the compiler may change. How the index range is split
across programs or launches is a mapping choice.

## Algorithm family and structure

Single-pass out-of-place reversal: a streaming copy whose read and write
address sequences run in opposite directions. No reduction, scan or sort is
involved. The mirroring may be applied on the read side (contiguous output
tiles reading descending input addresses), on the write side (contiguous
input tiles writing descending output addresses), or by reversing a tile on
chip between a contiguous load and a contiguous store.

## Precision and accumulation

No arithmetic is performed on the data; values are moved bit-exactly in the
input dtype, and the output dtype equals the input dtype. Masked or padded
lanes may hold any fill value because they are never stored.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

There is no cast, copy, packing or cached state. The input must be consumed
as-is; no host-side copy, view or conversion may precede the kernel, and
nothing may be cached across calls.

## Permitted implementation mappings

- Tile length, elements per program, number of programs, vector width,
  pipelining depth and launch geometry are free.
- The number of launches is free: the index range may be covered by one
  launch or partitioned across several, provided each element is still
  processed once and no intermediate is written to global memory.
- The read strategy: a masked load at mirrored addresses, a per-element
  gather at mirrored indices, or a contiguous tile load followed by an
  on-chip tile reversal; likewise on the write side.
- Tail handling by explicit masks, by gather padding values whose lanes are
  never stored, or by the DSL's bounds-clipped stores. Edge handling is
  required wherever the task's fixed `N` is not a multiple of the chosen
  tile; supporting shapes other than the task's declared shape is not
  required.
- Whether `N` is a runtime argument or a compile-time constant.

## Forbidden substitutions

- Computing the result with PyTorch (`torch.flip`, `input.flip(0)`,
  `torch.fliplr`/`flipud`, `torch.gather` or `torch.index_select` with a
  reversed index, `torch.take`, `out.copy_(...)`, slicing-based tricks)
  instead of a kernel.
- Returning `input`, a view of it, or a negative-stride view.
- Any dtype conversion.
- Reading or writing the data in more than one logical traversal, or
  staging through a scratch buffer in global memory, because either adds a
  full extra read and write of the array; this counts passes in the
  algorithm, not physical memory transactions.
- Mutating `input`.

## Permitted PyTorch operations

- `torch.empty_like(input)` for the output.
- Reading `input.numel()`, `input.shape`, `input.dtype`, `input.device`, and
  obtaining the current stream.

Everything else in `torch` is forbidden inside `run()`. In particular no
`torch.arange` index construction on the host: index arithmetic belongs in
the kernel.
