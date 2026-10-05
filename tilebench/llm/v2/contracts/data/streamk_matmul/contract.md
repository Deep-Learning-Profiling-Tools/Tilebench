# streamk_matmul: canonical algorithm contract

## Functional semantics

Dense matrix product `C = A @ B` with `A` of shape `(M, K)`, `B` of shape
`(K, N)` and `C` of shape `(M, N)`, all in the same dtype. This is the
reference `torch.matmul(a, b)` evaluated with TF32 tensor-core math enabled
for fp32 (the reference module sets `allow_tf32 = True`), so for fp32
operands the reference itself rounds operands to TF32. The verifier
tolerance is `atol 1.0, rtol 1e-2` (config `verify`). There is no
reference-side preprocessing.

## Inputs and outputs

The entry point is called as `run(a, b)`: `a`, `b` positional; no keyword
arguments are passed.

- `a`: `(M, K)`, contiguous row-major, dtype one of fp16, bf16, fp32.
  Read-only.
- `b`: `(K, N)`, contiguous row-major, same dtype as `a`. Read-only.
- Returned: one new tensor `(M, N)` of the input dtype, allocated inside
  `run()` on every call. It must not alias either input. For fp32 inputs the
  output buffer may itself serve as the fp32 accumulation target; for
  fp16/bf16 inputs the accumulation target is a separate fp32 buffer that is
  cast into the output at the end.

## Required logical stages

1. **Zero-fill** of the fp32 accumulation target `Cacc` (`(M, N)`, fp32):
   every call, inside `run()`, before stage 2. For fp32 inputs `Cacc` is the
   output; otherwise a separate fp32 buffer.
2. **Stream-K wave**: a persistent launch of `P` programs, where `P` is the
   number of streaming multiprocessors of the device at call time. The
   output tile grid has `T = ceil(M/tm) * ceil(N/tn)` logical tiles and each
   tile has `I = ceil(K/tk)` K-iterations. The Stream-K region consists of
   `T mod P` tiles, plus `P` further tiles when `T - (T mod P) > P` (at
   least two full waves remain). Its `S * I` K-iterations (S = region size)
   are divided into `P` contiguous ranges of equal length (the leading
   programs take the remainder). Each program walks its range segment by
   segment: a segment is the intersection of the range with one tile; the
   program accumulates that tile's K-slabs for the segment in a local fp32
   accumulator and then adds the partial result into `Cacc` with fp32
   atomic adds (masked to `M`, `N`). There is no fixup kernel, no semaphore
   or flag array and no partial-sum workspace: the zero-filled `Cacc` is the
   only combination target, and a tile cut across several programs is
   completed purely by the atomic adds.
3. **Data-parallel tiles**: the remaining `T - S` tiles, one tile per
   program, each reducing the full K range in a local fp32 accumulator and
   writing the tile to `Cacc` with ordinary (non-atomic) stores. These tiles
   are disjoint from the Stream-K region. Launched only when `T - S > 0`.
4. **Final cast** (fp16/bf16 only): copy `Cacc` into the output with one
   rounding to the output dtype. For fp32 inputs there is nothing to do.

Stages 2 and 3 both depend on stage 1 and write disjoint tiles, so they may
run in either order or be merged into one persistent launch; stage 4
depends on both. Tile ids are mapped to `(row-block, column-block)`
through a grouped (swizzled) order on both sides; the specific swizzle is a
locality choice.

## Algorithm family and structure

Hybrid Stream-K GEMM: a Stream-K region whose tile x K-iteration space is
split evenly across one persistent program per SM with atomic-add fixup into
a pre-zeroed fp32 `Cacc`, plus a data-parallel region of whole tiles with
plain stores. Per tile the K dimension is reduced slab by slab in fp32; for
Stream-K tiles the segments are combined by fp32 atomic adds, so the
combination order across segments is non-deterministic and those tiles are
not bitwise reproducible run to run (accepted within the tolerance). Full
tiles are reduced sequentially within one program. No sort or scan.

## Precision and accumulation

- Accumulation is fp32 for every input dtype; the accumulation target
  `Cacc` is always fp32.
- fp32 operands are rounded to TF32 for the tensor-core multiply (matching
  the reference); fp16/bf16 operands are multiplied natively.
- fp16/bf16 results are rounded to the output dtype exactly once, after all
  partials have been combined in fp32; never per partial.
- Out-of-range rows/columns of edge tiles read as zero for the operands and
  are never stored.

## Preprocessing and timing boundary

Inside `run()` and timed, on every call: the device SM-count query, the
output allocation, the zero-fill of `Cacc` (a memset of `M*N` fp32 values),
any descriptor or metadata construction, both launches, and the final cast
for half dtypes. `A` and `B` are delivered row-major as `(M, K)` and
`(K, N)`. Any layout transformation of an operand that an implementation
chooses to perform (for example a transposed copy of `B`) must be done
inside `run()` on every call and is timed; no state keyed on input identity
or contents may be cached across calls (see open review items).

## Permitted implementation mappings

- Logical tile shapes, K-slab width, swizzle grouping, pipelining depth,
  vector widths and all other launch parameters; whether `K` or the SM
  count are compile-time constants.
- Tensor-memory-accelerator descriptors versus pointer loads; a runtime
  `while` loop versus a static-trip-count loop for the K-slabs.
- Two launches (Stream-K wave, data-parallel tiles) in either order, or one
  persistent launch covering both regions.
- Host-side or device-side evaluation of the partition arithmetic.
- Using the output as `Cacc` for fp32 versus a separate fp32 buffer.

## Forbidden substitutions

- `torch.matmul`, `torch.mm`, `torch.addmm`, `torch.einsum`, the `@`
  operator, `F.linear`, or any BLAS / library GEMM.
- A pure data-parallel GEMM (no Stream-K region) or a split-K scheme with a
  separate reduction kernel or partials workspace.
- Combining Stream-K segments through semaphores, flags or an owner program
  instead of atomic adds into the zero-filled `Cacc`.
- Accumulating in fp16/bf16, or rounding half-dtype partials before the
  final combination; skipping TF32 rounding for fp32 operands is a
  different numeric mode and is not the canonical behaviour.
- Caching a transposed or repacked operand across calls; skipping the
  per-call zero-fill.
- Mutating `a` or `b`; returning a view of an input.

## Permitted PyTorch operations

- `torch.empty((M, N), ...)` for the output; `torch.zeros((M, N),
  dtype=torch.float32, ...)` for the separate fp32 `Cacc` of half dtypes;
  `Tensor.zero_` on the fp32 output used as `Cacc`.
- `Tensor.copy_` from `Cacc` into the half-dtype output as the final cast.
- `torch.cuda.get_device_properties(...).multi_processor_count` for the SM
  count; the current CUDA stream for launching.
- `.shape`, `.dtype`, `.device`, `Tensor.stride`, `Tensor.is_contiguous`,
  `Tensor.contiguous` as a no-op guard; `Tensor.t` / `Tensor.contiguous`
  only for an in-run, uncached operand re-layout as described above.

Everything else in `torch` is forbidden inside `run()`.

## Open review items

- Whether an operand layout transformation (a transposed copy of `B`) may
  be prepared once and reused across calls, outside the timed region, or
  must be performed inside `run()` on every call. Until resolved, this
  contract requires the in-run, uncached behaviour.
- Whether the bytes model should count the per-call zero-fill of `Cacc`,
  the atomic read-modify-write traffic of the Stream-K region and, for half
  dtypes, the fp32 round trip of the final cast, or remain the minimal
  one-read-one-write GEMM bound.
