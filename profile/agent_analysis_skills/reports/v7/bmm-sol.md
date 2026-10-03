# FP16 Batched Matmul, BATCH=32, M=640

TileLang takes **51.5 us**, versus **29.3 us Triton**, **31.8 us cuTile**, and
**23.3 us PyTorch** in the recorded autotuned CSV. Its latency is respectively
**1.758x, 1.619x, and 2.210x** these references. PyTorch is the fastest of every
available non-TileLang latency column; there is no PyTorch capture to explain
its advantage. These ratios use printed CSV values, not the stored speedup
columns (`results/csv/batched_matmul_autotune.csv:59`, `csv_checks.json`).

The best-supported operator-level explanation is **operand-staging and
synchronization dependencies that underfeed the tensor pipeline**. TileLang
uses lane-issued async global-to-shared transfers, explicitly synchronizes
threads after each tensor-memory GEMM, and uses two pipeline stages. Both DSL
references emit TMA loads/stores; Triton selects four stages. Machine code
confirms the staging and completion-wait structure, and counters show much
lower tensor activity for the same problem. This is a causal inference, not a
measured stall breakdown or a compiler defect attribution.

## Side-by-Side Evidence

| Quantity | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| CSV latency, us | 51.5 | 29.3 | 31.8 |
| NCU duration, ns | 49248 | 29952 | 29920 |
| Winning output tile | 128x128 | 128x128 | 128x128 |
| Winning K tile | 64 | 32 | 64 |
| Stages / occupancy hint | 2 stages | 4 stages | occupancy=4 |
| Grid CTAs | 800 | 800 | 800 |
| Threads / CTA | 128 | 128 | 256 |
| Registers / thread | 137 | 135 | 64 |
| Shared memory / CTA, bytes | 67584 | 66608 | 50452 |
| Register-limited CTAs / SM | 3 | 3 | 4 |
| Shared-memory-limited CTAs / SM | 3 | 3 | 4 |
| Achieved active warps, % | 6.127 | 15.754 | 28.207 |
| Tensor-pipeline activity, % elapsed | 16.022 | 27.065 | 26.454 |
| Dynamic warp instructions | 4011200 | 5448800 | 1521177 |
| DRAM read bytes | 31052288 | 28331264 | 25073664 |
| DRAM write bytes | 3795968 | 1683968 | 1592832 |

Every NCU entry refers to range **0**, action **0** of
`evidence/reports/targeted_<backend>.ncu-rep`. Exact records, units, and metric
identifiers are in `metrics.json`; the complete metric-name inventory and
kernel identity are in `inventory.json`. Row identifiers are
`gpu__time_duration.sum`, `launch__grid_size`, `launch__block_size`,
`launch__registers_per_thread`, `launch__shared_mem_per_block`,
`launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`,
`smsp__inst_executed.sum`, and `dram__bytes_{read,write}.sum`.

## Mechanism and Contradictions

All winners compute the same 800 full 128x128 output tiles, with group size 1.
The TileLang snapshot allocates shared A/B tiles and tensor-memory accumulation,
then executes `T.copy`, `T.gemm`, and `T.sync_threads` in the reduction loop
(`evidence/implementations/impl_tilelang.py:95-127`); warp specialization is
explicitly disabled at lines 74-76. Triton loads tensor descriptors and uses
`tl.dot` (lines 64-75 of its snapshot); its cached B transpose is visible at
lines 22-29. cuTile uses tile loads and MMA (lines 52-72 of its snapshot).
Thus the selected source contracts and staging differ despite matched output
geometry. No new configuration was compiled or searched.

The saved TileLang SASS contains **48 static `LDGSTS.E.BYPASS.128`** transfers,
no TMA instructions, and direct `STG.E.ENL2.256` stores. Triton has **10 static
`UTMALDG.3D`** and one `UTMASTG.3D`; cuTile has **30** and **two**. All three
have `UTCHMMA`, so tensor-core absence is contradicted. TileLang's SASS shows
tensor completion `UTCBAR`/`SYNCS.PHASECHK.TRANS64.TRYWAIT`, block synchronization,
then copy-group completion waits (`tilelang.sass.txt:325-345`, `:475-491`).
Triton's TMA and tensor instructions are visible at `triton.sass.txt:250-305`;
cuTile's at `cutile.sass.txt:234-239` and `:629-632`. These are static instruction
families, not executed counts. `sass_checks.json` has all per-kernel counts,
including NOPs, and no parsing warnings.

The measured consequence is tensor underuse: TileLang reaches 16.0% tensor
activity versus about 27% for both references. Multiplying each activity fraction
by its NCU duration gives approximately **7.89, 8.11, and 7.92 us** of averaged
tensor-active time. This derived comparison is consistent with similar tensor
work spread over a longer elapsed interval, not substantially more arithmetic.
Completion waits and staged operand movement provide a concrete dependency
path that can produce that behavior; their exact latency cost is unmeasured.

A pure resource-occupancy explanation fails against Triton: both have three-CTA
register/shared-memory limits and the same grid, whereas achieved warp activity
differs by 2.57x. Achieved activity is not theoretical occupancy. cuTile's four
CTAs and eight warps per CTA provide a larger residency ceiling, plausibly aiding
overlap, but this does not explain the Triton comparison by itself.
Nor does instruction count alone explain the result: TileLang executes **26.4%
fewer** warp instructions than Triton while taking **64.4% longer** in NCU.
Against cuTile it executes 2.64x as many, but that ratio is not a latency share.

TileLang moves more DRAM bytes: reads are 9.6% above Triton and 23.8% above cuTile;
writes are 2.25x and 2.38x. This is a credible secondary memory/cache difference,
but DRAM bytes do not establish uncoalescing, bandwidth saturation, or redundant
logical loads. Cache-level requests/sectors/hit rates and memory throughput were
not captured. Likewise, 800 CTAs on 148 SMs permit finite-wave effects, but matched
TileLang/Triton geometry and resource limits offer no independent evidence of a
different tail. A timeline would be needed to quantify either dependency stalls
or temporal imbalance. The larger static TileLang listing is not a diagnosis.

## Identity and Limits

The manifest, both autotune logs, and CSV agree on FP16, BATCH=32, M=640 and all
selected configurations. The config's FLOP formula and recorded size
16777216000 are consistent with N=K=M=640 (10 K iterations TileLang/cuTile,
20 Triton); explicit capture-time N/K tensor-shape records are absent.
Each report contains one matching `bmm_kernel` action on NVIDIA B200, sm_100,
with 148 SMs. Implementation snapshots match the supplied SHA256 catalog.
The supplied package catalog identifies TileLang 0.1.11, Triton 3.6.0, and
cuda-tile 1.3.0. The extraction uses the assigned PDF-era Python path. However,
capture-time PyTorch/CUDA runtime versions and GPU UUID are not independently
recorded in the supplied reports/manifest; NCU's path explicitly names CUDA
13.2 tooling, which does not prove the runtime version. Those remain provenance
limits rather than assumed verification.

NCU ratios are **1.644x versus Triton and 1.646x versus cuTile**, kept separate
from CSV ratios. The capture uses kernel replay, cache-control none, and a
single matching launch; benchmark configuration uses CUDA graphs and L2 flush.
NCU does not capture host overhead, PyTorch, or Triton's initial cached transpose.
Stall, transaction-efficiency, and temporal counters are absent from the checked
inventory. The available evidence supports an operator-level staging/dependency
explanation, but cannot isolate TMA, stage depth, explicit synchronization, or
epilogue effects into independent latency contributions. No compiler-source
attribution is made, so compiler tracing was not needed.

## Reproduction and Access

Run `source profile/env.sh` followed by `"$PY" output/extract.py` from this
worktree. It re-extracts raw numeric NCU records and original helper JSON and
copies the preimported SASS; its equality check passed for all **867** records.
Only assigned inputs were intentionally read. Outside-worktree execution/access
was limited to `/scratch/arustagi/tilebench_pdf_env` Python/runtime dependencies,
`/opt/nvidia/nsight-compute/2026.1.1/extras/python` NCU API and system libraries.
Shell startup automatically invoked Lmod under `/opt/ohpc/admin/lmod` and searched
Lua paths under `/usr/share/lua` and `/usr/lib64/lua`; it emitted a missing-posix
warning but did not prevent extraction. No bundle source, other worktree, history,
internet, profiling, benchmarking, or autotuning was accessed.
