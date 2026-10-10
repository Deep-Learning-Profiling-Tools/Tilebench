# cuTile Lowering: From `ct.*` to Tile IR

cuTile compiles in two parts. The open Python front end turns the kernel into
Tile IR and runs a few passes; the closed back end (`tileiras`, shipped with
the CUDA toolkit) turns Tile IR into machine code. The front-end IR shows
exactly what each `ct.*` call asks the back end to do. The back end's own
decisions (warp roles, tensor-core path, shared-memory staging, loop unrolling)
are visible only in SASS. Checked against cuda-tile 1.3.0 for `sm_100`.

## Get the IR

```bash
<python> <skill-dir>/scripts/cutile_ir.py <operators>/<op>/impl_cutile.py <kernel_fn> \
    --arg 10240x10240:float16 --arg 49:float16 --arg 10240x10240:float16 --arg 7 --arg 7 --arg 2 --arg 128 \
    --out <output>/cutile_<op>.tileir
```

`<operators>` is the checkout's operator directory: `benchmarks/operators` or
`tilebench/benchmarks/operators`, depending on the branch.

Arguments are the kernel's parameters in call order: arrays as `SHAPE:dtype`,
scalars and `ct.Constant` tile sizes as values (the winner config). The script
runs the front end only and builds the kernel signature from the shapes, so it
needs no GPU, driver or `tileiras`, builds no binary and launches nothing. It
describes each array as cuTile does for a contiguous, 16-byte-aligned array at
launch; a non-contiguous or sliced argument in the benchmark would need a
different signature. The mangled kernel name in a saved report ends with the same constants
(`…I7_I7_I2_I128`), which is how to confirm the specialisation matches.

## Reading Tile IR

### Front-end passes (`cuda/tile/_compile.py`, `_transform_ir`)

`eliminate_assign_ops` -> dead-code elimination -> dataflow analysis ->
`add_divby_pass` (emits `assume_div_by` / `assume_bounded` facts from the
arguments) -> `token_order_pass` (threads a `Token` through every memory
operation to fix their order) -> `rewrite_patterns` -> `hoist_loop_invariants`
-> `split_loops` -> dead-code elimination.

### `ct.gather` / `ct.scatter` are per-element pointer operations

`ct.gather(arr, idx, padding_value=v)` becomes, for every element of the tile:

```
$75 = tile_astype(x=$55)                      # index -> uint64
$79 = raw_cmp(lhs=$75, rhs=$78, fn="lt")      # bounds mask against the array extent
$82 = pointer_offset(pointer=$81, offset=$75)
$86, $87 = load_pointer(pointer=$82, mask=$79, padding_value=$85, token=…)
```

No contiguity information survives: the back end receives a tile of
independent pointers and a mask, even when the indices are consecutive or
shared across lanes. That is the origin of per-element bounds compares
(`ISETP.GE.U32`), index arithmetic and narrow loads/stores in SASS, and of an
instruction count per element far above peers. An explicit `ct.where(mask, …)`
on top of `padding_value` adds a second test.

Fingerprint: instruction-bound kernel (high issue rate, ALU pipe high,
`not_selected`/`math_pipe_throttle` stalls), identical DRAM bytes to peers, gap
shrinking as the element widens.
What to write instead: `ct.load` / `ct.store` on a partition view whenever the
accessed region is a regular tile (next rule); pad the array once instead of
masking every access.

### `ct.load` / `ct.store` are whole-tile operations

```
$110 = make_partition_view(array=out, tile_shape=(8192,), …)
$111 = tile_store(view=$110, index=($11), tile=$86, allow_tma=None, …)
```

The back end sees one tile with a known shape and stride and may use wide
accesses or TMA (`allow_tma`). Mixing a gathered tile with a tile store forces
the back end to re-map elements between the two ownership patterns, which
appears in SASS as shared-memory stores, a barrier and loads before the final
wide store.

### Loops and memory operations inside them

Python `for` loops over `ct.Constant` bounds stay loops in Tile IR
(`for $144 in range(…) (with acc.0 = …)`); whether each is unrolled is a
back-end choice (a 7x7 stencil has been seen with the inner tap loop unrolled
and the outer one kept as a backward branch).

`hoist_loop_invariants` moves pure arithmetic out of loops (a row bounds mask
ends up in the outer loop, the column mask in the inner), but memory operations
carry the ordering `Token` and are not hoisted. A scalar `ct.load` of a
coefficient inside a tap loop is therefore reloaded and re-converted every
iteration, for every thread.

What to write instead: load small read-only operands once before the loops
into a tile and index that tile.

### Thread mapping and reductions are back-end choices

Tile IR says only `Tile[dtype,(1,128)]` and `tile_reduce(xs=…, axis=1)`; it
does not say how many threads own the tile. The back end picks the CTA shape,
and it can pick one element per thread across several warps where Triton and
TileLang put the same tile in one warp. Every reduction then crosses warps
through shared memory: in-warp `SHFL`, `STS`, `BAR.SYNC`, `LDS`, combine,
`STS`, `BAR.SYNC`, `LDS`. Read `launch__block_size` and the shared-store count
from the report, and compare with sibling reports in the brief's all-reports
table. Some warps of a wide CTA may exit before later phases; per-PC execution
counts show which.

### Math is lowered literally

`a / b` on a tile is `raw_binary_arith fn="truediv"` and stays a checked IEEE
division per element (`MUFU.RCP`, Newton `FFMA` steps, `FCHK`, a guarded slow
path), even when `b` is one scalar broadcast to the tile. `ct.exp` uses the
accurate `expf` expansion (an `FFMA` range-reduction chain around `MUFU.EX2`).
What to write instead: compute the reciprocal of a scalar once and multiply.

### TMA can be chosen for plain tile loads and stores

`tile_load` / `tile_store` carry `allow_tma=None`, which leaves the choice to
the back end. When it picks TMA for a simple streaming tile, data is staged
through shared memory behind mbarriers (`UTMALDG`/`UTMASTG`, `SYNCS…TRYWAIT`,
`NANOSLEEP`), and the wait loops themselves execute a large number of
instructions. Shared memory per CTA rises and can cap resident CTAs.

### Hints and matmul

`occupancy` and similar hints are compiler options, not launch facts: check
`launch__occupancy_limit_*` in the report (a 222 KB shared-memory CTA runs one
per SM whatever the hint says). `ct.mma` appears as a single IR op; whether it
becomes `UTCHMMA` with tensor memory and role warps (`NANOSLEEP`, `SYNCS`,
`LDTM`/`STTM` per iteration) or legacy `HMMA` with register accumulators is
decided in the back end and must be read from SASS.

## Using This in a Report

Chain for a cuTile mechanism: source line -> Tile IR op (quote it) -> SASS
offset -> counter or PC samples. State plainly which part is verified in Tile
IR (what the front end emitted) and which is back-end behaviour read from SASS.
Name the source change that would alter the IR.
