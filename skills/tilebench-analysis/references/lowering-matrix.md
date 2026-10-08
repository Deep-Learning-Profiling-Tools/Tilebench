# The Lowering Decision Matrix

Every tile DSL compiler has to make the same decisions to turn a tile program
into a kernel. A performance gap between two backends on the same operator is
the sum of the decisions they made differently. This framework asks the same
twelve questions of every backend, answers each from that backend's intermediate
code, and explains the gap from the cells that differ.

Fill the matrix before writing mechanisms. It replaces "look at the generated
code" with a fixed set of things to find in it.

## How to Fill It

One column per backend, one row per decision. Each cell holds a short answer
and its evidence: the intermediate-code line or op (generated CUDA for
TileLang, `ttgir` for Triton, Tile IR for cuTile), or a SASS offset where the
decision is made below that level. Write `not determined` when the evidence
does not show it; never fill a cell from what the DSL usually does.

| # | Decision | Question to answer | TileLang: generated CUDA | Triton: `ttgir` | cuTile: Tile IR / SASS |
|---|---|---|---|---|---|
| 1 | Element ownership | Which elements does one thread own, and along which axis are a warp's lanes adjacent? | Index expressions in terms of `threadIdx.x` and the unrolled loop variable | `#ttg.blocked` `sizePerThread` / `threadsPerWarp` / `warpsPerCTA` / `order` | Tile shape and partition view; lane mapping from SASS addresses |
| 2 | Access width | How wide is each global load and store, and what stopped it being wider? | Vector type of the access (`half_t`, `int4`, `ulonglong4`), scalar when the index is not unit-stride or is guarded | Layout versus pointer order: contiguous per thread vectorises, otherwise per element | `tile_load`/`tile_store` (tile) versus `load_pointer`/scatter (per element) |
| 3 | Bounds handling | Is out-of-range access handled per element, hoisted, or avoided? | `condval` guards, tail `if` on the loop variable | `tt.load` mask and how it is computed | `raw_cmp` mask, `padding_value`, extra `where` |
| 4 | Specialization | Which shapes, strides and divisors are constants in the kernel? | Literals in the code | `tl.constexpr` constants versus runtime `i32` arguments | `typed_const`, `assume_*`, constants in the mangled name |
| 5 | Loop structure | Which loops are unrolled, which remain, and what is recomputed each iteration (including loads)? | `#pragma unroll` loops, bodies | `scf.for`, what LICM left inside | IR `for`; memory ops stay inside; unrolling from SASS |
| 6 | Staging and layout conversion | Does data pass through shared memory, and why? | Shared buffers, element-wise fills versus `T.copy` | `ttg.convert_layout`, `ttg.local_alloc` | SASS (`STS`/`BAR`/`LDS`), back-end decision |
| 7 | Arithmetic path | Where are values converted, are operations packed, which tensor-core path, and where do accumulators live? | Casts, `tcgen05…` versus `mma…` calls, fragment versus TMEM | `arith.extf` placement, `ttng.tmem_alloc` versus register MMA | `tile_astype`; MMA path from SASS |
| 8 | Async and synchronization | What is issued asynchronously, who issues it, and how many waits and barriers per loop iteration? | `cp_async…`, `__syncthreads()`, `mbar.wait`, `AllReduce` | `async_copy…`/TMA, stage indexing, waits in the loop | SASS (`SYNCS`, `NANOSLEEP`, `BAR`), back-end decision |
| 9 | Resource footprint | Registers, shared memory and resident CTAs: what in the code drives them? | `__launch_bounds__`, fragment sizes, unrolling; NCU launch stats | `.json` metadata, layouts; NCU launch stats | Hints versus NCU launch stats |
| 10 | Data footprint and reuse | How many bytes must one CTA fetch for the outputs it produces, and how much of that is shared with its own other warps? Derive expected L1 miss sectors from the tile shape (a 1-row stencil tile fetches 3 rows per output row; a 2-row tile fetches 4 for 2) | Tile shape and index expressions | Tile shape from constants and layout | Tile shape from constants |
| 11 | Formulation | How many passes over the tile does the algorithm make, how many reductions, and what does each combine step cost (one min/max instruction, or a compare-and-select chain on value and index pairs)? Is a result computed twice? | Reduction calls and their combiner in the code | Number of `tt.reduce` ops and the ops inside each combiner region | Number of `tile_reduce` ops and their combiner ops |
| 12 | Tuning surface | Which config values did the autotuner pick, and does changing them alter rows 1-11? | Config space versus what appears in the code | Same | Same; hints may be inert |

## From Matrix to Explanation

1. **Mark the rows where backends differ.** Identical rows explain nothing;
   skip them in the report.
2. **Account for the gap in samples first.** PC samples are time. The brief's
   "Where the extra time sits" table gives each slower backend's excess samples
   by stall reason. Whatever reason holds most of the excess is where the gap is
   spent, and the explanation has to land there. An instruction-count
   difference is only the cause if the excess sits in executing instructions
   (`not_selected`, pipe throttles, no stall recorded); if the excess is
   `long_scoreboard`, the gap is load waiting even when the slower backend also
   executes more instructions.
3. **Check the memory hierarchy.** Equal sectors and equal bytes mean equal
   traffic, not equal waiting. Read the brief's load order line for each
   kernel: loads issued before the first wait overlap, and a load issued after
   a wait is a second round trip per iteration. Also compare resident warps,
   since they set how many round trips overlap across the SM. For row 10 compare the
   expected miss sectors with `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum`
   and the L2-to-L1 traffic `l1tex__m_xbar2l1tex_read_bytes.sum` with the
   logical input size, per backend. Equal DRAM bytes do not mean equal traffic:
   L2 can absorb refetches that still cost an L1 miss each. Requested sectors
   and missed sectors answer different questions; look at both.
4. **Tag each differing cell as extra work or waiting.** Extra work raises the
   instruction ratio (rows 2-5, 7, 11: narrow accesses, per-element guards,
   runtime divisions, retained recomputation, extra converts, costlier
   combiners). Waiting lowers the issue rate (rows 1, 6, 8, 9, 10: few loads in
   flight, serialised loads, barriers, handshakes, bank conflicts, few resident
   warps). The time model in `brief.md` says how much of each gap is
   which; the tagged cells must account for it. If they do not, a row is wrong
   or missing.
5. **Confirm each differing cell in the measurements.** A cell is a claim about
   code; it becomes a mechanism when SASS execution counts, a counter or PC
   samples show its consequence. For an extra-work cell, check the arithmetic:
   the opcode families it implies must carry the excess in the dynamic opcode
   totals. If two backends use different instructions for the same step (an
   `FFMA` chain versus `FMUL` plus a compare), compare the sums, not one
   family. Do not assign an instruction excess to a cell whose opcodes do not
   add up to it.
6. **Attribute each differing cell** to one of:
   - *author*: the implementation chose this construct and another construct in
     the same DSL would change the cell;
   - *compiler*: the DSL lowered a reasonable construct this way and the author
     has no source-level way to change it;
   - *autotune*: another point in the search space changes the cell;
   - *capability*: the DSL cannot express or emit what a peer does.
   This attribution is the finding about the DSL. State the evidence for it,
   and say `not determined` between author and compiler when you cannot tell.
7. **Test the attribution against the scale trend.** The brief lists the winner
   config at every shape. If you attribute a gap to a config-specific effect
   (a hint, a tile size, a thread count), find neighbouring shapes where the
   winner differs: the gap should move with the config. If shapes with a
   different config show the same gap, the config is not the cause; say so and
   look again. Likewise, an argument that "another dtype's winner avoids X and
   is still slow" only holds if that winner really avoids X.
8. **Derive the fix from the cell.** The source-level change is whatever would
   move the differing cell to the faster backend's answer.

## In the Report

Include the matrix restricted to differing rows, with evidence in each cell.
Then one short paragraph per differing row that matters: what differs, work or
waiting, the measurement that confirms it, the attribution, the change. Keep
rows that differ but do not matter to a single line saying why not.
