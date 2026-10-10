# NCU Analysis for a TileBench Case

How to analyse the saved Nsight Compute reports of one operator across the
three backends. The reports already exist (no harness, no collection,
no PM-sampling timeline), there are three kernels to compare instead of one to
tune, and every finding has to be tied to SASS and then to the compiler's
intermediate code.

Run every dimension for every backend and write the answers side by side. Do
not stop at the first difference: most gaps are two or three patterns at once.

## Where the Numbers Are

`tb_case.py` writes, per backend, `ncu.json` (every metric of every kernel, as
`{"metrics": [{"metric": name, "value": v, "unit": u, "kernel": k, "range_index": r, "action_index": a}, ...]}`),
`annotated_sass_r?a?.txt` (offset, warp executions, PC samples, source line,
SASS, stall samples per instruction) and `detail_r?a?.json`. `brief.md` puts
the common metrics side by side. Anything else comes straight from the report:

```python
import ncu_report                       # PYTHONPATH=<nsight-compute>/extras/python
a = ncu_report.load_report(path).range_by_idx(0).action_by_idx(0)
names = set(a.metric_names())
a["smsp__inst_executed.sum"].value()                    # scalar metric
m = a["inst_executed"]; ids = m.correlation_ids()       # per-instruction metric
for i in range(m.num_instances()):
    pc = ids.as_uint64(i); count = m.as_uint64(i)
    sass = a.sass_by_pc(pc)                             # takes one PC
    src = a.source_info(pc)                             # .file_name(), .line(), may be None
a["smsp__pcsamp_warps_issue_stalled_long_scoreboard"]   # per-PC stall samples, same access
a["sass__inst_executed_per_opcode"]                     # dynamic opcode histogram
a.rule_results_as_dicts()                               # NCU's own rules (large; filter)
```

The saved reports do carry per-PC execution counts, per-PC stall samples and
source lines (TileLang's point into generated `tvm_kernels.cu`; match them to
the generated CUDA by content). A metric absent from `metric_names()` is
missing, not zero.

## The Six Dimensions, Per Backend

### 1. Launch geometry and occupancy

Metrics: `launch__grid_size`, `launch__block_size`, `launch__registers_per_thread`,
`launch__shared_mem_per_block`, `launch__occupancy_limit_{blocks,registers,shared_mem,warps}`,
`launch__waves_per_multiprocessor`, `sm__maximum_warps_per_active_cycle_pct`
(theoretical), `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved).

- CTAs per SM is the smallest occupancy limit; that limit names the cause
  (registers, shared memory, warps).
- `launch__shared_mem_per_block` reads 1,024 B above what the compiler
  allocated, and 1,024 B for a kernel that uses no shared memory. Subtract it
  before comparing with the IR, and do not read 1,024 B as staging.
- Waves below 1: the grid does not fill the GPU. Between 1 and 2: a partial
  last wave, so some CTAs wait for a slot. Above about 4: averaged out.
- Theoretical 100% with achieved far lower: stalls, not launch config; go to
  dimension 3.
- Compare: check grid x tile covers the same problem for all three, then work
  per CTA and per thread. Different tiles make raw totals misleading.

### 2. Block balance and tails

No timeline is saved. Use `sm__cycles_active.{avg,max,min}` spread, the
partial-wave arithmetic from dimension 1, NCU's imbalance rules, and per-PC
execution counts (a PC executed fewer times than its neighbours marks warps
that exited early or a loop with uneven trips). State a tail as inferred.

### 3. Stall reasons and per-instruction hotspots

Aggregate: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`.
Per instruction: `smsp__pcsamp_warps_issue_stalled_<reason>`, normalised by
`smsp__pcsamp_sample_count`.

| Reason | Waiting on | Usual cause |
|---|---|---|
| `long_scoreboard` | long-latency result | a global load that has not returned |
| `short_scoreboard` | short-latency result | shared or local memory, bank conflicts |
| `mio_throttle`, `lg_throttle` | memory-instruction queue full | many narrow loads/stores, saturated shared pipe |
| `barrier` | other warps at a sync | `__syncthreads`, cross-warp reductions |
| `wait` | fixed-latency pipe | `MUFU`, conversions, tensor ops |
| `math_pipe_throttle` | a compute pipe is full | genuinely compute-bound |
| `not_selected` | nothing: eligible, another warp issued | enough parallelism; a sign of instruction-bound code |
| `no_instructions`, `drain`, `dispatch_stall`, `branch_resolving` | launch, exit, control | short-lived warps, many CTAs, branches |

- A stall is charged to the instruction that needs the value, not the one that
  produced it. Walk back in the annotated SASS to the producing `LDG`/`LDS`.
- `long_scoreboard` is not only load waiting. A warp polling an mbarrier
  (`SYNCS...TRYWAIT` and the branch after it, `NANOSLEEP`) is recorded the same
  way. The brief's "at signal waits" column counts those samples; subtract them
  before calling a kernel latency-bound, then decide from the SASS which warp is
  waiting: an idle role warp costs nothing, a compute warp waiting for a TMA
  copy is waiting on memory by another route.
- Samples are taken at a fixed rate, so sample counts compare as time between
  backends. For each slower backend, find which reasons and which PCs hold its
  excess samples. That is where its gap is spent.
- Rough starting points only, to be confirmed in SASS: `long_scoreboard`
  holding the largest share of samples suggests memory latency;
  `short_scoreboard` prominent suggests shared memory or dependency chains;
  `barrier` prominent suggests synchronization. No fixed percentage decides it.

### 4. Pipes and tensor cores

Metrics: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
(nonzero for both MMA and tcgen05 kernels), `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`
(nonzero only on the tcgen05/TMEM path; it can read higher or lower than the
first, so neither bounds the other, and a comparison must use the same one for
every backend), `sm__inst_executed_pipe_tensor_subpipe_{hmma,imma,dmma}...`
(legacy MMA forms), `sm__pipe_{alu,fma}_cycles_active...`,
`sm__inst_executed_pipe_{xu,fma,alu}.avg.pct_of_peak_sustained_active`,
`sm__throughput...`, `smsp__issue_active.avg.per_cycle_active`,
`smsp__warps_eligible.avg.per_cycle_active`.

- Eligible warps below 1 with low issue rate: nothing to issue, the kernel is
  waiting. Eligible warps well above 1 with issue rate near its ceiling:
  instruction-bound.
- Issue-slot floor: `smsp__inst_executed.sum` / number of SM sub-partitions is
  the least cycles the kernel can take. Sub-partitions =
  `device__attribute_multiprocessor_count` x `device__attribute_num_schedulers_per_multiprocessor`
  (148 x 4 = 592 on B200). If a backend's floor already exceeds a
  peer's measured `smsp__cycles_active.avg`, no amount of occupancy or memory
  tuning closes the gap; it executes too much.
- Matmul-shaped work with tensor pipe at 0 is a missed path; read which MMA
  form each backend emitted from SASS (`UTCHMMA`, `HMMA`). A high tensor pipe
  percentage is not a good sign by itself: a kernel on the legacy `HMMA` path
  can show the busiest tensor pipe and be the slowest, because it spends more
  pipe time on the same arithmetic. Compare pipe-busy time (percentage x
  duration) between backends, and check `sm__pipe_tc_cycles_active` to see who
  is on the tcgen05 path.

### 5. Binding resource

One line per backend naming what limits it: DRAM bandwidth
(`gpu__dram_throughput...` near 80% or more), load latency (`long_scoreboard`
dominant with DRAM well below peak), instruction issue, one pipe, the shared
memory pipe (`l1tex__throughput` high with DRAM low), synchronization, or
occupancy. Two backends with different binding resources respond to different
changes, and a cost hidden under one bound can appear on other hardware or
another dtype where the bound moves.

### 6. Memory access and cache behaviour

Metrics: `dram__bytes_{read,write}.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
and `..._lookup_miss.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`,
`l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`,
`l1tex__m_xbar2l1tex_read_bytes.sum`, `lts__t_sector_hit_rate.pct`,
`smsp__sass_inst_executed_op_tma_{ld,st}.sum`,
`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum`,
`l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum`,
`smsp__sass_average_data_bytes_per_sector_mem_global_op_{ld,st}.ratio`,
`smsp__sass_inst_executed_op_{global,shared,local}_{ld,st}.sum`,
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`,
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum`.

- A request is one warp-level load and a sector is 32 bytes, so a fully
  coalesced warp load takes lanes x bytes per lane / 32 sectors: 16 for a
  128-bit load, 32 for a 256-bit load, 2 for a 16-bit load. More than that
  means the lanes are spread out. Bytes per sector tops out at 32; lower means
  lanes use part of each sector they touch.
- Data moved by TMA (`UTMALDG`/`UTMASTG`) does not appear in the global
  load/store instruction, sector or hit-rate counters, which read zero for a
  TMA kernel. Use the `tma` counters above.
- Requested sectors, missed sectors and DRAM bytes answer different questions.
  More missed sectors than a peer with equal DRAM bytes means the same data is
  refetched from L2 (overlapping footprints between CTAs).
- Equal sectors and bytes do not mean equal waiting. In the annotated SASS,
  count the loads a thread issues before the first instruction that consumes
  one: loads issued together overlap, a load issued after a consumer has
  waited is another round trip.
- Shared memory: use the bank-conflict counters directly. Wavefronts per
  access are not a conflict measure on their own, because a wide access needs
  several with no conflict.
  Bank = byte address / 4 mod 32; evaluate the index expression for one warp to
  confirm a conflict.
- Local loads or stores above zero are register spills.
- `smsp__sass_inst_executed_op_global_ld.sum` does not count 256-bit
  `LDG.E.ENL2.256` loads; use requests or sectors.
- It does not count async copies (`LDGSTS`, from `cp_async`) either. Their
  count is `smsp__inst_executed_op_ldgsts.sum`; a kernel that loads only this
  way shows zero global loads beside nonzero global load sectors, and its wait
  is the `DEPBAR` after the copies, not the instructions between them.

## MI300X: rocprof-compute Instead of NCU

The MI300X bundle has no `.ncu-rep`. `brief.md` carries a rocprof-compute
section and the bundle holds `<backend>/amd_metrics.json` (every counter) and
`<backend>/sampled_asm_<kernel>.txt`. Only Triton runs on MI300X, so the
comparison is against PyTorch on the same GPU or against Triton's own B200 or
GH200 case, not between DSLs.

What carries over, with AMD names (a wavefront is 64 work-items; a workgroup is
the block; LDS is shared memory):

| Dimension | Read from |
|---|---|
| Launch geometry | `Wavefront / Wavefront Launch Stats / {Grid Size, Workgroup Size, VGPRs, SGPRs, LDS Allocation}` |
| Occupancy and utilization | `System Speed-of-Light / {Wavefront Occupancy, CU Utilization, IPC, VALU Active Threads}` |
| Stalls and hotspots | stalled samples by reason and the sampled instructions: `WAITCNT` (waiting for memory, on `s_waitcnt vmcnt`), `BARRIER_WAIT` (`s_barrier`), `ARBITER_NOT_WIN` (ready, another wavefront issued), `ALU_DEPENDENCY` |
| Pipes | `System Speed-of-Light / {VALU, SALU, VMEM, MFMA, Branch} Utilization` |
| Memory | `System Speed-of-Light / {vL1D Cache Hit Rate, L2 Cache Hit Rate, L2-Fabric Read BW, L2-Fabric Read Latency, LDS Bank Conflicts/Access}` |

What does not carry over:

- No execution counts. Only instructions that received a sample are listed, so
  instruction totals, the issue-slot floor and per-element work ledgers cannot
  be computed. Get the full instruction stream from
  `triton_ir.py --hardware MI300X` (`.amdgcn`) and match by offset and source line.
- Sample counts can be small (tens for a short kernel). Treat a split of a few
  samples as indicative only and say how many samples it rests on.
- The NCU metric names, the Blackwell SASS glossary and the pattern table
  below are NVIDIA-specific. Use them as a list of things to look for, not as
  names to cite.

## Patterns, With Their SASS and IR Evidence

For each pattern that the dimensions show, cite the SASS that proves it and
then find the cause in that backend's intermediate code. The last column is the
row of the [lowering matrix](lowering-matrix.md) that usually produces it.

| Pattern | NCU signals | SASS evidence | Where the IR shows the cause |
|---|---|---|---|
| Grid too small, partial wave | waves below 1, or just above 1 with a tight occupancy limit | registers per thread, block size | Tile size and thread count (rows 9, 12) |
| Uncoalesced or narrow global access | sectors/request high, bytes/sector low, `lg_throttle`, many `LDG` per warp | `LDG.E.U8/U16` per element next to a peer's `LDG.E.128` | Ownership and access width (rows 1, 2): `sizePerThread`, vector type, `load_pointer` versus `tile_load` |
| Latency-bound | `long_scoreboard` dominant, DRAM well below peak, low eligible warps | one load then a wait per iteration; samples on the first consumer | Loop structure and loads in flight (rows 5, 8, 10): loads inside a retained loop, one small tile per warp |
| DRAM-bound | DRAM throughput near peak in every backend, parity | all loads issued up front | Nothing to fix; other differences are hidden under it |
| Instruction-bound | high issue rate, `not_selected`/`math_pipe_throttle` hold the excess samples, a pipe near peak, DRAM low | long per-element sequences: bounds compares, divides, conversions, guards | Bounds, specialization, arithmetic path, formulation (rows 3, 4, 7, 11) |
| Refetched data | missed sectors and L2-to-L1 bytes above a peer, equal DRAM bytes | same addresses loaded by neighbouring CTAs | Data footprint (row 10): tile shape versus stencil reach |
| Shared-memory bank conflicts | bank-conflict counters well above zero relative to shared accesses, `short_scoreboard` + `mio_throttle`, `l1tex__throughput` high | byte or strided `STS`/`LDS` | Staging (row 6): element-wise shared fills, shared-to-shared moves, the destination index |
| Layout conversion through shared memory | shared stores and loads with one barrier in an elementwise kernel | `STS`, `BAR.SYNC`, `LDS` between load and store | Row 6: `ttg.convert_layout`, gather mixed with tile store |
| Synchronization | `barrier` samples, `BAR` executions per iteration | `BAR.SYNC` and the `LDS` after it | Row 8: `AllReduce`, `__syncthreads`, cross-warp `tt.reduce` |
| Occupancy capped | theoretical occupancy below 100, a register or shared-memory limit | register count, spills (`LDL`/`STL`) | Row 9: unrolling, fragments, `__launch_bounds__`, hints |
| Tensor path | tensor pipe zero on matmul work, or tcgen05 pipe zero while a peer's is not, or the pipe idle much of the kernel | `UTCHMMA` versus `HMMA`, `NANOSLEEP`/`SYNCS` | Row 7 and 8: TMEM versus register MMA, TMA versus lane copies |
| Divergence | `smsp__thread_inst_executed_per_inst_executed.ratio` below 32 | predicated tails, per-warp early exits | Rows 3, 5: tail guards |

## Comparing the Three

1. Confirm the three reports are the same problem: bytes read, grid x tile,
   winner config against the launch stats and the mangled kernel name.
2. Put the six dimensions side by side and mark every difference, including
   ones that favour the slower backend.
3. Account for each gap in samples by stall reason and PC, then by executed
   instructions (dynamic opcode totals; sum instruction families that do the
   same job before comparing).
4. Check whether a difference actually costs time: a backend can execute more
   instructions at no cost when both are waiting on memory, and an occupancy
   cap costs nothing when a pipe is already saturated. The sibling reports
   (other dtypes, other GPU) are the test: find one where the suspected factor
   is absent or different and see whether the gap moves.
5. Treat each NCU rule as a hypothesis; rules misfire on packed instructions
   and on memory-bound kernels ("low utilization").
