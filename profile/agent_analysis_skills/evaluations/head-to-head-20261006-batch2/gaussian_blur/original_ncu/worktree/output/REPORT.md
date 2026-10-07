# B200 Gaussian Blur: Maximum FP32 Case

This saved-evidence comparison covers a 10240 x 10240 FP32 image and a 7 x 7 filter.
TileLang is fastest in the benchmark CSV, with a modest advantage over Triton and cuTile.
The strongest causal evidence is reduced on-chip load/sector and instruction demand, despite TileLang's severe register occupancy limit.
No GPU execution, tuning, recapture, installation, or network access was performed.

## Benchmark Result and Selected Configurations

| Backend | CSV latency (ms) | Selected winner | CSV latency / TileLang |
|---|---:|---|---:|
| TileLang | 1.1436 | BLOCK_R=4, BLOCK_C=256, threads=128 | 1.0000 |
| Triton | 1.2487 | BLOCK_R=8, BLOCK_C=64, num_warps=8 | 1.0919 |
| cuTile | 1.2713 | tile_r=2, tile_c=128, occupancy=4 | 1.1117 |

Source: `results/B200/csv/gaussian_blur_autotune.csv`; raw winners: `evidence/gaussian_blur{,_tilelang}_autotune.json`.
Triton is the fastest non-TileLang backend. TileLang reduces its latency by 8.42%, within the study's +/-10% parity band.
Against cuTile, TileLang reduces latency by 10.04%, just across that threshold; Triton and cuTile differ by only 1.81%.
The percentages use rounded CSV latency columns; do not infer statistical significance from this one maximum-size case.

## Capture Correspondence

All three report SHA-256 values match `evidence/reports.json`; `hash_verification.json` preserves the checks.
That manifest names dataset revision `21037737b7e371d38f3d029367dc3967d5b31a23`, while the supplied source pin is `17d2d4f6` and archive pin is `9455c0bd`.
Those identifiers are different provenance objects, not proof of identical capture and current source; no history was consulted.
Each report contains one range with one action, indexed (0,0); exact kernel names are in `action_inventory.json` and every metric record.
Triton's embedded `impl_triton.py` kernel body equals the supplied body; its wrapper lacks the current `warmup=1, rep=3` tuner arguments.
cuTile's embedded kernel body also equals the supplied body; the only file difference is the `CutileAutotuner` import namespace.
Capture grids TileLang=(2560,40), Triton=(1280,160), cuTile=(5120,80), and block sizes 128/256/128 agree with the logged maximum-case tiles.
cuTile's mangled kernel includes FP32 and `I7_I7_I2_I128`; its occupancy hint is not independently recoverable from that name.
TileLang records `/tmp/tmprph76jf1/tvm_kernels.cu` line mappings but no CUDA content: byte-identical source/config correspondence cannot be proved.
Its SASS has FP32 scalar loads/FFMA, bound constants 0x2800 (10240), and tile offsets compatible with the supplied specialization.
These are strong structural matches, with residual uncertainty about capture compiler/package versions and TileLang's exact generated source.

## NCU Side-by-Side

All entries below come from range 0/action 0; raw values, units, and exact names are in `selected_records.json`.
Million-count entries divide their raw counters by 1e6. Duration is `gpu__time_duration.sum` divided by 1e6.

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Captured duration (ms) | 1.134976 | 1.274624 | 1.279904 |
| Registers/thread (`launch__registers_per_thread`) | 254 | 32 | 48 |
| Theoretical occupancy (`sm__maximum_warps_per_active_cycle_pct`) | 12.50% | 100.00% | 62.50% |
| Achieved occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 11.09% | 86.25% | 59.14% |
| Waves/SM (`launch__waves_per_multiprocessor`) | 345.95 | 172.97 | 276.76 |
| L1/TEX throughput (`l1tex__throughput.avg.pct_of_peak_sustained_elapsed`) | 64.78% | 98.45% | 76.26% |
| DRAM throughput (`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`) | 9.89% | 8.79% | 8.79% |
| Global load L1 hit rate (`l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`) | 95.65% | 97.58% | 93.16% |
| Global load instructions, millions (`smsp__sass_inst_executed_op_global_ld.sum`) | 180.607 | 206.417 | 240.818 |
| Global load sectors, millions (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`) | 799.388 | 1180.529 | 859.599 |
| Useful load bytes/sector (`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio`) | 25.80 | 17.68 | 24.27 |
| Executed warp instructions, millions (`smsp__inst_executed.sum`) | 555.008 | 748.749 | 1125.581 |
| ALU elapsed-cycle activity (`sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed`) | 25.07% | 31.72% | 71.72% |
| LG throttle / issue-active (`smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio`) | 0.664 | 13.317 | 0.021 |
| Long scoreboard / issue-active (`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`) | 0.628 | 5.153 | 2.633 |
| Math throttle / issue-active (`smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio`) | 0.042 | 0.190 | 2.393 |

The capture ratios Triton/TileLang=1.1230 and cuTile/TileLang=1.1277 exceed the CSV ratios; causal counters are not replacement benchmark timings.
Stall ratios are normalized by issue-active cycles, not percentages, and different resident-warp counts affect their magnitude.

## Causal Diagnosis

**Triton: inefficient lane mapping and L1 load-queue saturation.** In `impl_triton.py:23-37`, each output performs masked stencil loads and coefficient loads.
SASS maps columns using `IMAD.SHL ... 0x2` and `LOP3 ... 0x3e`, yielding every-other-column lanes for scalar loads, plus overlapping wider loads.
Its 17.68 useful bytes/sector and 1180.529 million sectors explain why high L1 hit rate still coexists with 98.45% L1/TEX throughput.
PC sampling locates 12138 LG-throttle samples at the coefficient `LDG.E` on embedded source line 36, and 5184 at an input `LDG.E` on line 34.
This implicates frequent coefficient loads as well as input accesses; it is not solely an image-load bandwidth problem.
The NCU rule engine independently flags LG throttle and uncoalesced global accesses (`rules_triton.json`).

**TileLang: larger work per block and instruction-level parallelism trade registers for less issued work.** `impl_tilelang.py:53-62` fully unrolls taps into a register fragment.
The selected block produces 1024 outputs versus Triton's 512 and cuTile's 256; TileLang SASS stages many independent loads before the FFMA sequence.
It executes fewer global loads and total warp instructions, and moves fewer load sectors than either competitor, reducing demand on L1/TEX and issue resources.
The cost is 254 registers/thread, a register limit of two blocks/SM, and 12.5% theoretical occupancy; zero local load/store instructions show no measured spill.
Its hottest long-scoreboard PC is an FFMA with 10445 samples mapped to generated CUDA line 40, consistent with waiting on staged-load operands.
This occupancy limit is a remaining constraint, not an explanation for being slower: TileLang is fastest here despite it. More occupancy alone is not a causal fix.

**cuTile: gather/address code and retained row-loop overhead.** `impl_cutile.py:38-46` uses a row loop, padded gather, and scalar coefficient load; SASS retains a backward branch.
Repeated `ISETP`, `LEA`, and address operations accompany gathers and coefficient loads; 1125.581 million instructions and 71.72% ALU activity support this overhead mechanism.
The hottest math-throttle examples include `R2UR` (4089 samples) and gather-bound `ISETP.GE.U32.AND` (2723), rather than only accumulation arithmetic.
SASS uses paired FP32 `FFMA2`; the rule engine's non-fused-FP32 warning misses that form and should not motivate an accumulation rewrite.
cuTile's very low LG throttle distinguishes it from Triton; similar latency does not imply the same bottleneck.

## Exclusions, Confidence, and Next Directions

DRAM reads are approximately one image (419.53-420.60 MB) across all backends, with low DRAM utilization; the main gap concerns on-chip work, not redundant HBM traffic.
Tensor-pipe activity is zero for all three; this direct scalar stencil does not demonstrate a backend-specific missing-MMA explanation.
Large wave counts exclude small-grid starvation. Per-SM active-cycle min/max are close to their means; no `pmsampling:` metrics were collected, so a utilization timeline or tail shape cannot be claimed.
No local-memory loads/stores occur in any capture; TileLang and Triton execute no shared-memory instructions, excluding shared-bank conflicts there.
For future implementation work, prioritize Triton's lane/coefficient load mapping, then cuTile's gather checks and row-loop lowering; preserve TileLang's load overlap while reducing live values.
These are evidence-ranked directions, not tested speedup promises or permission to sweep configurations; this diagnosis compares the logged winners only.
Artifacts: `analyze_saved.py`, `comparison.json`, `hotspots_*.json`, `sass_*.txt`, embedded source inventories, complete scalar dumps, `selected_records.json`, and `COMMANDS.md`.
