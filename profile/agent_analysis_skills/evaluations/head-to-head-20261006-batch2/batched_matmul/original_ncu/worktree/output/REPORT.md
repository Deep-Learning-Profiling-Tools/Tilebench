# B200 Batched Matmul FP16: Selected-Winner Diagnosis

## Scope and result

This saved-evidence study covers BATCH=32, M=N=K=640, FP16 inputs/output, and FP32 accumulation.
The primary performance source is `results/B200/csv/batched_matmul_autotune.csv`, row `M=640,fp16`.
TileLang is 1.754x slower than Triton and 1.616x slower than cuTile; Triton/cuTile differ by only 8.53%, within the study's parity band.
The strongest supported explanation is tensor-core underuse associated with TileLang's collective copy/MMA/wait schedule and low achieved residency; synchronization/pipeline latency is the leading causal hypothesis, not a measured stall decomposition.

| Benchmark result | TileLang | Triton | cuTile | Torch |
|---|---:|---:|---:|---:|
| CSV latency, microseconds | 51.4 | 29.3 | 31.8 | 23.3 |
| TileLang latency / backend latency | 1.000x | 1.754x | 1.616x | 2.206x |

Torch is the fastest non-TileLang backend overall; Triton is the fastest of the three profiled DSLs. No Torch capture is supplied, so its mechanism is outside this diagnosis.
The arithmetic workload is 16,777,216,000 FLOPs; each complete FP16 matrix tensor contains 26,214,400 bytes, derived from the shape.

## Winners and capture/source correspondence

| Selected configuration | TileLang | Triton | cuTile |
|---|---|---|---|
| Output tile M x N | 128 x 128 | 128 x 128 | 128 x 128 |
| K tile / software stages | 64 / 2 | 32 / 4 | 64 / compiler-managed |
| Threads or compiler hint | 128 threads | 4 warps | occupancy=4 hint |
| Group size | 1 | 1 | 1 |

Both supplied winner logs independently agree on TileLang's configuration; `selected_case.json` preserves the exact selected entries and CSV row.
The supplied source pin is `17d2d4f6`, archive pin `9455c0bd`; `evidence/reports.json` labels the reports with dataset revision `21037737b7e371d38f3d029367dc3967d5b31a23`. These identifiers are not interchangeable.
All report SHA-256 hashes match that manifest. Each contains one range and one action; exact kernel names and report identities are in `capture_inventory.json` and every metric record.
Triton's embedded `bmm_kernel`, descriptors, transpose cache, and block-size hook match the current source by AST. Whole-file differences add the CDNA3 default fallback and change the untuned wrapper's config lookup; they do not change this B200 kernel body.
cuTile's embedded kernel and run wrapper match by AST; the whole-file difference is the `CutileAutotuner` import namespace. Its mangled kernel encodes K_TILES=10, tile=128x128x64, GRID_M=GRID_N=5, and GROUP_SIZE=1.
TileLang embeds compiler headers but no content for `/tmp/tmpftr2k34b/tvm_kernels.cu`. Its PTX has stride 640 and its launch fits the winner; copy/MMA/wait instructions agree with the current TMEM path, but exact generated-source equality and the full winning-config replay cannot be independently proven.
Embedded Triton/cuTile paths come from another capture environment. Source correspondence does not establish identical compiler versions, runtime arguments, cache state, or benchmark timing windows; no original capture command or harness is supplied here.

## Side-by-side NCU evidence

All entries below are action 0 in range 0 of `evidence/<backend>_fp16.ncu-rep`; `selected_records.json` retains the exact metric, value, unit, backend, report, and kernel.

| Exact NCU metric (units) | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `gpu__time_duration.sum` (ns) | 48,576 | 32,064 | 34,848 |
| `launch__grid_size` (blocks) | 800 | 800 | 800 |
| `launch__block_size` (threads) | 128 | 128 | 256 |
| `launch__waves_per_multiprocessor` | 1.802 | 1.802 | 1.351 |
| `launch__registers_per_thread` | 137 | 135 | 64 |
| `launch__shared_mem_per_block` (bytes) | 67,584 | 66,608 | 50,452 |
| `sm__maximum_warps_per_active_cycle_pct` (%) | 18.75 | 18.75 | 50.00 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (%) | 6.12 | 15.31 | 27.89 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (%) | 16.13 | 24.63 | 22.13 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` (%) | 19.84 | 32.46 | 25.99 |
| `dram__bytes_read.sum` (bytes) | 29,588,224 | 52,454,656 | 52,470,784 |
| `dram__bytes_write.sum` (bytes) | 3,688,192 | 51,616,000 | 50,128,640 |

NCU TileLang/Triton=1.515x and TileLang/cuTile=1.394x, derived from `gpu__time_duration.sum`; these differ materially from the CSV ratios and do not replace them.

## Mechanisms and alternatives

**Tensor-core underuse, not tensor-core absence.** All three SASS listings contain `UTCHMMA` and TMEM operations; TileLang PTX explicitly uses `tcgen05.mma...kind::f16`.
TileLang's lower tensor active percentage and SM throughput confirm underuse. They measure activity, not achieved FLOPs or an upper bound on a proposed optimization's speedup.

**Collective synchronization and limited overlap are the leading TileLang mechanism.** Current `impl_tilelang.py:78` disables warp specialization, and lines 118-130 stage copies, issue TMEM GEMM, synchronize inside the K loop, then drain TMEM to a register fragment and store.
Captured SASS uses `LDGSTS.E.BYPASS.128`/`LDGDEPBAR`/`DEPBAR` for copies, followed by `UTCHMMA`, `UTCBAR`, `SYNCS.PHASECHK...TRYWAIT`, and block synchronization before subsequent work; see `sass_tilelang.txt:223` and `:291` through `:325`.
TileLang has asynchronous copies and a real pipeline; the limitation is the collective wait schedule's opportunity to overlap future producer work with MMA, not an absence of pipelining.
Triton's `impl_triton.py:88-97` lowers descriptor loads/stores to `UTMALDG.3D`/`UTMASTG.3D`; cuTile's `impl_cutile.py:52-72` also lowers to TMA with TMEM MMA. Both avoid TileLang's per-thread copy/addressing path.
The captured TileLang headers provide async MMA arrival and thread-sync fence primitives (`embedded_tilelang_tcgen_05.h.txt:49-85`), while the current kernel explicitly requests block sync; this identifies a concrete scheduling boundary without attributing it to an uninspected compiler pass.
TileLang lacks collected stall counters and PC samples, so neither barrier dominance nor a quantitative copy-versus-MMA wait cost is established.

**Resource limits amplify latency, but register count alone does not explain the gap.** `launch__occupancy_limit_registers` and `launch__occupancy_limit_shared_mem` both allow 3 blocks/SM for TileLang/Triton and 4 for cuTile.
TileLang and Triton thus share the same theoretical occupancy despite their different achieved occupancy. The latter is a residency symptom, not proof that stalled resident warps disappear or that a particular resource caused the loss.
cuTile has higher theoretical/achieved occupancy but remains in parity with Triton, showing that occupancy alone does not determine runtime.

**Short-grid/tail sensitivity is secondary.** All reports identify NVIDIA B200 with 148 SMs; geometry matches the source's 5x5x32 TileLang/Triton grid and 25x32x1 cuTile grid.
No output tile or K tail exists for these divisible dimensions, and all CTAs have uniform logical work. The modest waves/SM permit a scheduling tail, but identical TileLang/Triton geometry cannot by itself explain their difference.
No PM-sampling metrics were collected, so no utilization timeline or measured tail duration is claimed. Static SM active-cycle extrema cannot replace a timeline.

**Faster DSLs still wait on async operations.** Triton has 1,090/1,673 long-scoreboard samples (65.15%); cuTile has 1,750/1,944 (90.02%), from `smsp__pcsamp_warps_issue_stalled_long_scoreboard` / `smsp__pcsamp_sample_count`.
Triton's largest long-scoreboard PC has 486 samples at `0x7feb1de9c260`, a branch after `SYNCS...TRYWAIT`; cuTile's largest has 683 at `0x7f9f2fa03d50`, a `NANOSLEEP.SYNCS` wait mapped to its store. Instance indices/correlation IDs are preserved in `selected_records.json`.
These PC locations identify waiting on synchronization/completion paths; labeling them ordinary uncoalesced global loads from the stall name alone would be unsupported.
Triton/cuTile DRAM read and write percentages are 21.36/21.01% and 19.65/18.77% of peak (`dram__bytes_{read,write}.sum.pct_of_peak_sustained_elapsed`), supporting latency rather than saturated HBM bandwidth in those captures.

**Memory traffic is not directly comparable.** TileLang's measured DRAM writes are below one complete output tensor; the other captures' writes approach two outputs. DRAM counters reflect cache/writeback/replay behavior, not just logical stores.
These differences prevent claiming a TileLang traffic advantage or applying one shared cold-cache roofline; cache-state equivalence is unproven. TileLang has no L2, coalescing, bank-conflict, or spill counters here.
Triton/cuTile have zero local-load/store sectors (`l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`); their shared-load bank-conflict counts are 1,282/5,759, but neither observation establishes TileLang's behavior.

## Ranked directions and confidence

1. Highest priority: examine the selected TileLang kernel's copy/MMA completion schedule and producer/consumer separation, preserving its winner as the baseline. Expected benefit is improved overlap and tensor utilization; magnitude cannot be estimated from this partial capture.
2. Next: examine TMEM-to-register epilogue live ranges and staging resource footprint. Lower resource demand may increase residency, but matching Triton's register count alone is unlikely to eliminate the measured gap.
3. Treat partial-wave scheduling and cuTile's completion waits as secondary opportunities; the Triton/cuTile benchmark difference is below the chosen divergence threshold.

High confidence: CSV ranking, selected log records, captured tensor-core use/underuse, and embedded Triton/cuTile kernel correspondence. Moderate confidence: TileLang collective scheduling is a material contributor. Unresolved: exact residency cause, TileLang stall shares, version/cache equivalence, and causal time attribution.
`COMMANDS.md` reproduces the offline extractions; `all_*.json`, `rules_*.json`, `stall_hotspots_*.json`, source inventories, and PTX/SASS mappings retain the supporting evidence. No GPU execution, autotune, recapture, install, network, or operator-source edits were performed.
