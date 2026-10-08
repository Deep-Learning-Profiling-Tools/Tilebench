# TileLang Lowering: From DSL Construct to Generated Code

NCU and SASS show what ran. This reference explains why TileLang emitted it:
which DSL construct and compiler rule produced a pattern, and what the kernel
author could write instead. Everything here was checked against TileLang 0.1.11
(generated CUDA for TileBench kernels, and the compiler source at that tag).
Lowering changes between versions, so confirm against generated code for the
case in hand before citing a rule.

## Get the Generated Code First

Saved TileLang reports embed `tvm_kernels.cu` with empty content, so the report
alone never shows the generated kernel. Regenerate it; nothing runs on a GPU:

```bash
<python> <skill-dir>/scripts/tl_codegen.py <checkout>/tilebench/benchmarks/operators/<op>/impl_tilelang.py <kernel_fn> \
    --tensor 20000000:float16 --tensor 20000000:float16 \
    --kw dtype=float16 --kw BLOCK_SIZE=1024 --kw threads=128 --out <output>/tilelang_<op>.cu
```

The script targets the hardware given with `--hardware` (default B200) and
needs neither a GPU nor `nvcc`. Operators that pick a kernel variant by GPU
architecture get the variant for that hardware. If an operator builds its
kernel inside a factory function, so there is no module-level kernel to name,
write a few lines of Python that obtain the kernel the same way the file's
`run()` does and compile it with CPU tensors of the right shape and dtype.

Tensors are the kernel's positional arguments in order (`ROWSxCOLS:dtype`);
`--kw` carries shape scalars, dtype strings and the winner config from
`brief.md`. Read the body after `__launch_bounds__`; it is usually under 40
lines. Generate the contrast dtype too and diff the two files.

**Check that it matches the capture.** Compare the generated structure with the
captured SASS (iterations per thread, load/store widths, guards). If they
disagree, the capture came from a different TileLang build; say so, and treat
the generated code as describing the installed version only. Example seen:
installed 0.1.11 emits one 16-byte load/store per thread for an int8 scatter,
while a released capture of the same kernel shows four 4-byte chunks.


### TIR before and after the passes

TileLang is built on TVM, so between the Python source and the generated CUDA
there is TIR. `tl_codegen.py` can write both ends of TileLang's pass pipeline:

```bash
<python> <skill-dir>/scripts/tl_codegen.py <impl_tilelang.py> <kernel_fn> --tensor ... --kw ... \
    --out <output>/tl.cu --tir <output>/tl.tir --lowered-tir <output>/tl.lowered.tir
```

- `--tir`: the kernel as written, with shapes and config bound: `T.copy`,
  `T.parallel`, `T.Pipelined`, fragment and shared allocations at their logical
  size.
- `--lowered-tir`: the device function after every pass, the form the CUDA is
  printed from: per-thread buffer sizes, explicit `threadIdx` index
  expressions, unrolled loops, inserted guards and intrinsic calls.

Use the CUDA for what the final code does. Use the two TIR files when the
question is which construct turned into a given piece of CUDA, or whether a
source construct survived at all (a `T.Pipelined` loop with no async copies in
the lowered form was not pipelined). The files show the pipeline's input and
output, not which individual pass made a change.

## Rules

### `T.Parallel` assigns elements to threads in strided chunks

Source: `src/transform/loop_partition.cc`, `LoopPartitioner::Partition`. With
`v` the vector width in elements and `t` the thread count, flattened element
`f` goes to thread `(f / v) % t` in iteration `f / (v * t)`:

```c
for (int i = 0; i < 8; ++i)            // BLOCK_SIZE / (threads * v) iterations, #pragma unroll
  output[blockIdx.x*1024 + i*128 + threadIdx.x] = ...   // v = 1, t = 128
```

- Adjacent threads own adjacent chunks, so each warp-level access is contiguous
  and coalesced (full bytes/sector).
- One thread's chunks are `v * t` elements apart. They can never be merged into
  a wider access, so stores stay `v` elements wide however many a thread owns.
- A 2D `T.Parallel(R, C)` is flattened row-major first, then split the same way.

Fingerprint: per-thread SASS accesses at offsets stepping by `t * v * sizeof`,
one narrow store per iteration, sectors per request healthy.
What to write instead: nothing in the loop body changes this; wider access
needs a larger `v` (next rule) or a `T.copy` of a contiguous region.

### Vector width is a bit budget reduced by every access in the body

Source: `src/transform/loop_vectorize.cc`. The planner starts at 128 bits (256
when the target and all memory accesses allow) and takes the GCD with what each
buffer access and the loop extent permit. It drops to scalar when:

- an index is not unit-stride increasing in the loop variable (a reversed index
  such as `input[n - 1 - idx]`, or a transposed one such as `A[base + j, i]`);
- the access sits under a data-dependent or per-element condition, including
  the guard inserted for `T.annotate_safe_value`;
- the loop extent is not a compile-time constant.

Seen: reversed copy -> scalar `LDG.E.U16`/`STG.E.U16`; indexed scatter whose
index is constant across the vector -> one 16-byte (int8) or 32-byte (fp32)
access per thread; guarded stencil taps -> scalar loads.
What to write instead: make the hot access unit-stride (pre-transpose or
pre-reverse the operand outside the kernel, as the Triton implementations do),
or split interior from border so the interior loop carries no guard.

### Out-of-range protection becomes per-access guards

Two sources of `if` in generated code:

- **Tail guard.** When the shape is not a multiple of the tile, stores are
  wrapped in a bound test that simplifies to something like
  `if (blockIdx.x*4 + (i >> 1) < 78125)`. The guard changes every few
  iterations, so the unrolled body compiles into groups (here pairs): loads of
  the next group are not issued until the current group's stores are.
  Fingerprint: guarded load/store groups in SASS, `long_scoreboard` samples on
  the first store of each group, few loads in flight.
- **Safe-value guard.** `src/transform/legalize_safe_memory_access.cc` wraps a
  global load whose index it cannot prove in range as
  `condval = in_bounds ? buf[idx] : fallback`. This happens whether or not the
  source calls `T.annotate_safe_value`; the annotation only chooses the
  fallback value (zero otherwise). A stencil gets one multi-term
  predicate per tap per output (7x7 taps x 8 outputs = 392 guarded scalar loads
  per thread). Fingerprint: predicated `@P LDG`, `ISETP` chains, register
  zeroing for the fallback value.

What to write instead: pad the input once or handle borders in a separate
kernel/branch so the interior needs no guard; choose shapes or tiles that
divide evenly when that is an option.

### `T.unroll` plus fragment accumulators makes straight-line code with many live values

A fragment of shape `(R, C)` on `t` threads gives each thread `R*C/t` scalar
accumulators (`float acc[8]`). Unrolled tap loops then emit taps x outputs
independent loads per thread, and nvcc keeps many of them live.

Every TileLang kernel is declared `__launch_bounds__(threads, 1)`. The second
argument tells nvcc that only one block per SM must fit, so it is free to spend
up to 255 registers per thread. This is what permits the 250+ register,
2-blocks-per-SM kernels; nothing in the DSL source asks for it.

Fingerprint: `launch__registers_per_thread` near 255, register-limited
occupancy near 12%, no `LDL`/`STL`.
What to write instead: fewer outputs per thread (smaller tile or more threads),
or a non-unrolled tap loop. Whether that wins is a measurement, not a given.

### Half precision is converted at every load

For fp16 inputs with a float32 fragment, the generated code is the fp32 kernel
plus `(float)input[...]` on every load and `(half_t)acc[...]` on every store
(`cuda_fp16.hpp`, `HADD2.F32` in SASS). Structure, guards and accumulator count
are identical to fp32. A gap that exists only in fp16 therefore comes from the
convert-per-load path or from how nvcc schedules it, not from a different
algorithm; diff the fp16 and fp32 SASS around the loads.

### Element-wise fill of a shared tile is not `T.copy`

`shared[i, j] = T.cast(G[base + j, col + i], dtype)` inside `T.Parallel` is
lowered to per-thread `tl::cp_async_gs<4>` calls: one 4-byte async copy per
element, 16 per thread per operand per iteration in the 32x64 case, because the
transposed index is not unit-stride so the 16-byte form cannot be used. A
`__syncthreads()` follows in each pipeline iteration.

Fingerprint: `LDGSTS` (or global-load) count equal to elements moved, about one
L1 sector per lane, `lg_throttle` present, tensor pipe starved; peers show
`UTMALDG` and no per-element loads.
What to write instead: give the kernel an operand whose GEMM tile is contiguous
(transpose once outside the kernel) and use `T.copy(G[...], shared)`.

### `T.reduce_sum` across threads is a block-wide reduction per output element

Reducing a fragment over a dimension that spans threads emits, inside an
unrolled loop over the outputs each thread owns:

```c
__syncthreads();
for (int i = 0; i < 8; ++i)
  tile_sum[i] = tl::AllReduce<tl::SumOp, 128, 4, 0, tl::NamedBarrier<128>>::run(tile_sum[i], workspace);
```

That is one shared-memory, barrier-synchronised reduction per owned element
(8 here), repeated every loop iteration (`src/tl_templates/cuda/reduce.h`).
Fingerprint: barrier stall dominant, `BAR`/`STS`/`LDS`/`SHFL` hot, time per
iteration well above a peer that accumulates in registers.
What to write instead: accumulate per thread across the loop and reduce once
after it, or lay the tile out so the reduced dimension stays within a thread.

The same declaration matters in moderate cases too: a kernel at 48 registers
fits 10 CTAs per SM where a 32-register peer fits 16 and runs the whole grid in
one wave.

A reduction that stays inside one warp (`AllReduce<Op, 32, …>` with 32 threads)
is shuffle-only and has no barrier, like the peers' single-warp reductions.

### Math

`T.exp` is nvcc's accurate `expf` (an `FFMA` range-reduction chain around
`MUFU.EX2`). Division is emitted as written, so an author-hoisted
`coef = 1 / x` followed by a multiply is one reciprocal per thread.

### Other verified behaviours

- **Warp specialization is disabled** by pass config in TileBench GEMM kernels
  (`TL_DISABLE_WARP_SPECIALIZED`), so copy, MMA and waits share threads.
- **fp32 GEMM uses the legacy MMA path** (`HMMA…TF32`, register accumulators);
  the tcgen05 path is taken for half-precision inputs.
- **`CumSum1D` scans on one warp** while the rest wait at a barrier.
- **`T.Pipelined(num_stages=n)`** rotates `n` shared buffers and issues the
  prologue copies before the loop; the steady-state body still contains a
  `__syncthreads()` per iteration.

## Using This in a Report

For each TileLang mechanism, give the chain in this order: DSL line -> the rule
above -> generated C (quote the two or three lines) -> SASS offset -> counter or
PC samples. State the alternative the author could write and the measurement
that would confirm it. Where generated code and captured SASS disagree, report
the version mismatch instead of forcing them together.

Triton and cuTile: their reports embed the Python kernel source with line
mapping, so the DSL line is direct, but this skill has no verified lowering
rules for their compilers. Describe their lowering from source plus SASS and
label it as inferred.
