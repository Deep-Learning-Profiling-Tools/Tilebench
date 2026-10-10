# Triton Lowering: From `tl.*` to Generated Code

Triton records what its compiler decided at four levels for every kernel:
`ttir` (the program as written), `ttgir` (after GPU decisions: layouts, shared
memory, tensor-core and pipelining choices), `llir` and `ptx`. `ttgir` is the
level that explains performance. Checked against Triton 3.6.0 targeting
`cuda:100`; confirm on the IR of the case in hand, since passes change between
versions.

## Get the IR

Compile the kernel for the profiled target from its source. This needs no GPU
and launches nothing:

```bash
<python> <skill-dir>/scripts/triton_ir.py <operators>/relu/impl_triton.py relu_kernel \
    --arg x_ptr=ptr:i8 --arg output_ptr=ptr:i8 --arg n_elements=int:20971520 --arg BLOCK_SIZE=const:2048 \
    --num-warps 2 --hardware B200 --out <output>/triton_ir
```

`<operators>` is the checkout's operator directory: `benchmarks/operators` or
`tilebench/benchmarks/operators`, depending on the branch.

Give one `--arg` per kernel parameter, in any order:

- `ptr:<dtype>` for a tensor pointer (`i8`, `i32`, `i64`, `fp8e4nv`, `fp16`, `bf16`, `fp32`, or the long names).
- `desc:<dtype>:<AxB>` for a `TensorDescriptor` argument with that block shape.
- `int:<value>` for a runtime integer. Pass the real value from the benchmark
  shape: Triton records whether it is divisible by 16 and turns a value of
  exactly 1 into a constant (`backends/compiler.py`, `get_int_specialization`),
  and both change the IR. A wrong value can silently give a different layout
  (for example `sizePerThread = [1]` and scalar loads instead of 16-byte loads).
- `float:<value>` for a runtime float, `const:<json>` for a `tl.constexpr`.

`--num-warps`, `--num-stages` and the constexpr values come from the winner
config. The script writes `<kernel>.{ttir,ttgir,llir,ptx,json}`: `ttir` is the
program before the GPU passes and `ttgir` after them. Always check the result
against the captured SASS (load widths, loads per thread, shared memory in the
launch stats) before citing it. The `#loc` lines map every IR value to a source
line, the same mapping NCU shows in the report.

To try a source variant (a different `sem`/`scope` on an atomic, a constexpr
instead of a runtime shape), copy the implementation under the output
directory, edit it, and compile the copy the same way. That shows what the
compiler would emit; it is not a measurement.

## Reading `ttgir`

The rules below were checked against the 148 Triton kernels of the B200
profiles (Triton 3.6.0): each states how many kernels show it. They describe
this version and target; confirm in the case's own IR before citing one.

### Layout decides which elements a thread owns

Each tensor type carries a layout, most often

```
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

`sizePerThread` consecutive elements belong to one thread, then the next lane,
then the next warp; a tensor larger than the product wraps, so the thread owns
several such runs. The `Coalesce` pass picks the layout so that a memory access
is contiguous per thread, up to 16 bytes: no kernel in the set has a global
load wider than 128 bits (`ld.global.v4.b32`, `LDG.E.128`). Operations
connected by data flow share one layout unless a conversion is inserted.

Consequences seen:

- **One access vectorises, the other pays for it.** Reversed copy: layout
  `sizePerThread = [8]` (8 fp16 = 16 bytes) makes the store one `STG.E.128` per
  run, but the load uses the same layout with a descending pointer, so it stays
  one `LDG.E.U16` per element and adjacent lanes are 8 elements apart. Each
  warp-level load then spans 16 sectors and uses 4 bytes of each: many L1 sector
  requests, high hit rate, `lg_throttle` on the loads, unchanged DRAM bytes.
- **2D tiles:** `sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA =
  [8, 1]` means two adjacent columns per thread and one row per warp. The lane
  stride along a row is 2 elements, which is what sets bytes/sector.

Fingerprint: per-thread wide access on one side, per-element narrow access on
the other, in the same kernel. What to write instead: make the expensive access
the contiguous one (pre-reverse or pre-transpose outside the kernel), or accept
the trade when the wide side is the store.

### `ttg.convert_layout` is a trip through shared memory

When two connected operations need different layouts (a reduction feeding a
store, a dot result feeding an elementwise op, a transposed store), the
compiler inserts `ttg.convert_layout`. Unless the two layouts differ only
within a thread, it lowers to shared-memory stores, a barrier and loads
(`STS…`/`BAR.SYNC`/`LDS…` in SASS). `RemoveLayoutConversions` deletes the ones
it can; those that remain are usually real cost: of the 51 kernels that keep a
`convert_layout`, 45 have shared stores and a barrier in their PTX, 4 a barrier
only, and 2 neither (a within-thread conversion). Count them
(`triton_ir.py` prints the count) and find their `#loc` source line.

### Masks and loads

`tl.load(ptr, mask=m, other=v)` stays one `tt.load %ptr, %mask, %other`: a
predicated load with a zero-fill, straight-line with no branch, so many loads
can be outstanding per warp. The mask and pointer arithmetic are tensor ops
evaluated per element (`arith.cmpi`, `arith.andi`, `tt.addptr`): separable row
and column tests are recomputed for every element unless written to broadcast
late. Shapes passed as runtime scalars keep `//` and `%` as real divisions;
`tl.constexpr` folds them. Read this from the PTX, not `ttgir`: a division by a
constant is still `arith.divsi`/`remsi` in `ttgir` and is turned into a
multiply and shift later (15 of 15 kernels), while a runtime divisor stays
`div`/`rem` in PTX (22 of 22).

A scalar loaded inside a Python loop (a stencil coefficient per tap) becomes a
`tt.load` plus `tt.broadcast` per iteration: every thread reloads it.

### Reductions

`tl.max`, `tl.sum`, `tl.argmax` are `tt.reduce` with an explicit combiner
region. With one warp per CTA (`warpsPerCTA = [1]`) a reduction is register
shuffles only (`shfl.sync.bfly`, `redux.sync` in PTX; `SHFL`, `CREDUX` in
SASS): all 5 one-warp reduction kernels are shuffle-only. With more warps it
also goes through shared memory and barriers (35 of 37; the exceptions reduce
along an axis that stays inside a thread or a warp). A
paired reduction such as argmax carries a compare/select combiner
(`FSETP`/`FSEL`/`ISETP`/`SEL` per shuffle step). Choosing `num_warps` larger
than the tile needs therefore adds shared-memory traffic; the all-reports
table in the brief shows it when one sibling report was tuned that way.

### Math

`tl.exp` becomes `llvm.nvvm.ex2.approx.f` on a scaled argument (`FMUL`, range
`FSETP`, `MUFU.EX2`). A division by a value that is uniform across the tile is
`arith.divf` / `div.full` per element in the IR, yet the back end can emit one
reciprocal per warp and a multiply per element; confirm from `MUFU.RCP`
execution counts instead of assuming either.

### Mixed precision follows the source expression

`acc += x * w` with fp16 `x`, `w` and fp32 `acc` is `arith.mulf` on f16 followed
by `arith.extf`: the product is formed in half precision (`HMUL2` on packed
pairs) and only then widened. Casting the operands first gives an fp32 multiply
and one convert per operand. This changes both cost and rounding.

### Matmul

`tl.dot` is rewritten by `AccelerateMatmul`. In the IR look for:

- `ttng.tmem_alloc` and `ttng.tc_gen5_mma` (accumulator in tensor memory, the
  tcgen05 path; `UTCHMMA` in SASS, `UTCQMMA` for fp8) versus a `tt.dot` left
  with an MMA-layout register accumulator (`mma.sync` in PTX; `HMMA`/`IMMA` in
  SASS). On B200, 20 of the 23 dot kernels take the tcgen05 path. The three
  that do not are `matmul_int8`, `block_sparse_attention` fp16 and the
  `kv_gemm` kernel of `linear_self_attention` fp32.
- `ttg.local_alloc` with `#ttg.nvmma_shared<{swizzlingByteWidth…}>`: operand
  staging buffers; `ttg.memdesc_index` indexes the pipeline stage.
- TMA: tensors loaded through `TensorDescriptor` appear as
  `ttng.async_tma_copy_global_to_local` (`UTMALDG` in SASS; 17 kernels). Plain
  pointer loads into a dot become `ttg.async_copy_global_to_local` +
  `ttg.async_commit_group` + `ttg.async_wait` (lane-issued async copies,
  `LDGSTS`/`DEPBAR` in SASS; the fp32 convolutions). The fp16 convolutions use
  neither: ordinary loads and shared stores feed the MMA.
- `num_stages` is realised by the `Pipeline` pass after `AssignLatencies` and
  `ScheduleLoops`: count the stage buffers and waits inside `scf.for`.
- An f32 `tl.dot` goes through `F32DotTC` (TF32 on tensor cores). On the
  tcgen05 path the SASS shows plain `UTCHMMA`; the `tf32` kind is visible in the
  PTX. On the register path it is `HMMA.1688.F32.TF32`.

### Pass order (from `triton/backends/nvidia/compiler.py`, `make_ttgir`)

Common prefix: `convert_to_ttgpuir` -> `coalesce` -> `f32_dot_tc` -> `plan_cta`
-> `remove_layout_conversions` -> `optimize_thread_locality` ->
`accelerate_matmul` -> `remove_layout_conversions` -> `optimize_dot_operands` ->
`optimize_descriptor_encoding` -> `loop_aware_cse`.

Then the pipeline branches on the target:

- Hopper and Ampere-class (`cuda:80`, `cuda:90`, so GH200): `fuse_nested_loops`
  -> LICM -> `combine_tensor_select_and_if` -> `hopper_warpspec` ->
  `assign_latencies` -> `schedule_loops` -> `pipeline`.
- Blackwell (`cuda:100`, so B200): `fuse_nested_loops` -> LICM ->
  `optimize_accumulator_init` -> `hoist_tmem_alloc` -> `promote_lhs_to_tmem` ->
  `assign_latencies` -> `schedule_loops` -> `warp_specialize` -> `pipeline` ->
  `optimize_partition_warps` -> `combine_tensor_select_and_if` ->
  `hoist_tmem_alloc` -> `remove_tmem_tokens`.

Common suffix: `loop_aware_cse` -> `prefetch` -> `optimize_dot_operands` ->
`coalesce_async_copy` -> `optimize_tmem_layouts` -> `tma_lowering` (Hopper and
later) -> `remove_layout_conversions` -> `interleave_tmem` ->
`reduce_data_duplication` -> `reorder_instructions` -> `fence_insertion` ->
`lower_mma`.

The order is copied from the source. What each pass does is read from its name
and its effect on the IR, not from the pass sources; diff `ttir` against
`ttgir` to see what happened to this kernel. The GH200 and B200 branches differ,
so a rule about tensor memory or warp specialization on one does not carry to
the other.

## Using This in a Report

Chain for a Triton mechanism: source line -> the `ttgir` construct (quote the
layout or op) -> SASS offset -> counter or PC samples. Name the layout, any
remaining `convert_layout`, the matmul path and the pipelining actually present
in the IR, and state the source change that would alter them.
