# jacobi_stencil_2d: canonical algorithm contract

## Functional semantics
One out-of-place Jacobi sweep of the four-point (von Neumann, no centre
term) stencil on a two-dimensional grid IN of shape (rows, cols):

    interior (1 <= r <= rows-2, 1 <= c <= cols-2):
        OUT[r, c] = 0.25 * (IN[r-1, c] + IN[r+1, c] + IN[r, c-1] + IN[r, c+1])
    boundary (first/last row, first/last column):
        OUT[r, c] = IN[r, c]

The centre value does not enter the average. The neighbours are summed
up, down, left, right and the sum is then scaled by 0.25; verification is
tolerance-based, so bit-exact association is not required.

## Inputs and outputs
- IN: shape (rows, cols), row-major contiguous, dtype fp16 | bf16 | fp32.
  Read-only; it must never be written.
- rows, cols: Python ints (second and third positional arguments). The
  task's declared shape is square (cols == rows); supporting shapes other
  than the declared one is not required.
- OUT: a freshly allocated tensor of shape (rows, cols) and dtype IN.dtype,
  allocated inside the entry point. Every element, boundary included, is
  written by the implementation's own device pass. No aliasing.
- The entry point takes (IN, rows, cols) positionally; no keyword arguments
  are passed.

## Required logical stages
1. Output allocation (uninitialised).
2. Stencil sweep: for each logical output tile, obtain the four neighbours
   of IN, form the interior average, select the untouched centre value on
   boundary positions, and store the tile.
Stage 2 depends on stage 1. There is exactly one sweep (one iteration); it is
a single logical stage and one launch suffices. The boundary copy and the
interior update belong to the same logical pass, in which every element of
OUT is written once from IN: a copy of the whole of IN into OUT (on the host
or by a device pass) followed by an interior-only kernel is not permitted,
because it adds a second traversal of the grid that the canonical sweep does
not have. Writing the boundary and the interior positions in one launch or
in separate launches over disjoint positions is a mapping choice.

## Algorithm family and structure
Single-iteration, out-of-place stencil (true Jacobi: every output value is
computed from the input grid only, never from already-updated outputs). No
reduction, scan or sort. Out-of-range neighbour reads must not contribute to
any stored value, either because they are masked or because the boundary
select discards them; stores are clipped to (rows, cols).

## Precision and accumulation
Arithmetic in the input dtype is the expected behaviour. Promoting the
four-term sum to fp32 and casting the result once to IN.dtype is permitted;
computing in a dtype narrower than the input is not. Output dtype equals the
input dtype.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

No halo padding, cloning or layout transform of IN on the host; no
cross-call caching of anything.

## Permitted implementation mappings
Tile shape, grid shape and order, launch parameters; neighbour access via
several shifted loads or gathers, a halo-extended tile load, or on-chip
staging; select-based versus two-region writes of the boundary, in the same
launch as the interior or in a separate launch over the disjoint boundary
positions; rows/cols as runtime or compile-time arguments; whether strides
are passed explicitly.

## Forbidden substitutions
torch.nn.functional.conv2d or any convolution / unfold / roll / shift
primitive; PyTorch slicing arithmetic (IN[0:rows-2, 1:cols-1] + ...) that
produces the interior; IN.clone(), torch.clone or copy_ to obtain the
boundary (a second full traversal of the grid); more than one sweep;
in-place update of IN.

## Permitted PyTorch operations
- torch.empty_like(IN), or torch.empty with IN's shape, dtype and device, for
  OUT.
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
