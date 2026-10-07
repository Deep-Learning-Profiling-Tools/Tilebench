# B200 Gaussian Blur FP32: Selected-Winner Comparison

At 10240 x 10240, TileLang records 1.1436 ms, Triton 1.2487 ms, and cuTile 1.2713 ms.
The strongest supported explanation is a tradeoff in per-thread work amortization,
cache-side transaction pressure, and bounds/address overhead, rather than HBM traffic.
TileLang wins while paying a substantial register-residency cost.
Against the fastest non-TileLang implementation, the CSV gap is within the study's
10% parity band; this is a modest local advantage, not a broad backend conclusion.

## Matched Work and Timing

CSV source: `results/B200/csv/gaussian_blur_autotune.csv`, line 41, unique
`params=input_rows=10240,dtype=fp32` row; exact ratios are in `csv_comparison.json`.
All current kernels implement the same direct 7 x 7 zero-padded stencil with FP32
accumulation, 104,857,600 outputs and 10,276,044,800 nominal floating-point operations.
Each implementation has one launch; the logical footprint is 838,860,800 bytes plus 196 coefficient bytes.

| Quantity | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| CSV latency, ms | 1.1436 | 1.2487 | 1.2713 |
| CSV latency / TileLang | 1 | 1.0919 | 1.1117 |
| Saved NCU duration, ms | 1.134976 | 1.274624 | 1.279904 |
| Output tile | 4 x 256 | 8 x 64 | 2 x 128 |
| Winner launch setting | 128 threads | 8 warps | occupancy hint 4 |
| Captured block threads | 128 | 256 | 128 |
| Captured grid (x,y) | (2560,40) | (1280,160) | (5120,80) |
| Outputs/thread, derived | 8 | 2 | 2 |
| Registers/thread | 254 | 32 | 48 |
| Register residency limit, blocks/SM | 2 | 8 | 10 |
| Achieved active warps, % of peak | 11.09 | 86.25 | 59.14 |
| Dynamic warp instructions, million | 555.008 | 748.749 | 1125.581 |
| Global-load requests, million | 180.607 | 206.417 | 240.818 |
| Global-load L1 sectors, million | 799.388 | 1180.529 | 859.599 |
| L1 throughput, % of sustained peak elapsed | 64.78 | 98.45 | 76.26 |
| HBM read + write bytes, GB (decimal) | 0.86079 | 0.85992 | 0.86277 |
| HBM throughput, % of sustained peak elapsed | 9.89 | 8.79 | 8.79 |
| ALU pipe instructions, % of peak elapsed | 25.07 | 31.72 | 71.72 |
| Eligible warps/scheduler active cycle | 0.569 | 2.815 | 3.417 |

TileLang reduces CSV latency 8.42% versus Triton and 10.04% versus cuTile.
Triton and cuTile differ only 1.81%, also parity. NCU ratios are separately
1.1230 and 1.1277 for Triton/TileLang and cuTile/TileLang; they support the same
ordering but do not replace the benchmark ratios or establish uncertainty bounds.

## Why the Selected Implementations Differ

TileLang's larger per-thread output bundle amortizes coefficient loads and indexing.
Its SASS prologue maps thread IDs to consecutive columns, uses constant shape bounds
and precomputed offsets, and retains many loaded values before their FFMA consumers.
Triton's prologue maps adjacent-column pairs per lane, then mixes scalar and 64-bit
loads; cuTile gathers paired rows while checking dynamic tensor bounds inside a row loop.
These are captured code differences, not conclusions drawn from source syntax alone.
Across the same output work, TileLang executes 25.87% fewer warp instructions than
Triton and 50.69% fewer than cuTile. Dynamic ISETP executions are 64.307, 129.434,
and 363.725 million respectively; IMAD executions are 20.070, 80.282, and 81.920 million.
The source/configuration and instruction mix support reduced bounds/address overhead.

Triton's high occupancy does not eliminate its cache-side transaction pressure:
it processes 47.68% more L1 load sectors than TileLang, with L1 throughput near
sustained peak and a load-queue-throttle ratio of 13.317 versus TileLang's 0.664.
This is consistent with lane layout, repeated weight loads, and overlapping widened
stencil loads increasing cache/LSU work. It is not evidence of 47.68% more HBM traffic.
All HBM totals are close; their low peak percentages reject HBM bandwidth saturation
as the principal contrast. High cache hit rates alone would not remove load-issue costs.

cuTile has a beneficial paired-FP32 arithmetic path: 80.282 million FFMA2 warp
executions versus 160.563 million scalar FFMA executions for each other backend.
That arithmetic saving coexists with a runtime outer-row loop and repeated gather
bounds/index work. Its SASS epilogue converts the compute layout through STSM,
BAR.SYNC, LDS.64, and STG.E.64. ALU activity and total issue activity (77.73%)
support overhead/pipe pressure despite efficient arithmetic; the store conversion
is an additional observed cost whose elapsed-time contribution is not isolated.

The counterargument is TileLang's register footprint: two possible resident blocks
give a 12.5% theoretical warp ceiling, and achieved activity is only 11.09%.
No local load/store sectors are recorded, so the high allocation did not cause
observed local-memory spill traffic. Fewer ready warps leave less dependency hiding,
offsetting the instruction/transaction savings and plausibly explaining why the
latency gain is much smaller than the instruction-count gain. No time shares are inferred.
Stall ratios above use the exact `per_issue_active.ratio` denominator, not wall-time %.

## Capture Correspondence and Limits

All report hashes match `evidence/reports.json` at dataset revision
`21037737b7e371d38f3d029367dc3967d5b31a23`; every report contains one range/action.
Device attributes identify NVIDIA B200, capability 10.0, and 148 SMs.
The supplied current source is pinned to `17d2d4f6`, winner archive to `9455c0bd`.
Triton/cuTile embedded kernel ASTs match current source. Whole files differ only
in Triton autotune warmup/rep arguments and cuTile's import path, respectively.
Grid/block geometry agrees with all logged winners; cuTile's name encodes FP32 and 7,7,2,128, but its occupancy hint is log-only evidence.
TileLang's embedded CUDA file has empty contents; its correspondence is inferred
from geometry and SASS constants (including 10240 bounds and tile shifts), not a
verified capture-time kernel body or compiler revision. Stack/cache/replay command
metadata is not supplied; the reader's 2026.1.1 version is not a capture-stack proof.
Thus these profiles support the implementation tradeoff, not a compiler defect claim.
The smallest discriminating evidence would be capture manifests tying stack/config
to each report and temporal attribution of cache pressure versus exposed dependencies.
No new GPU execution or collection was performed. Exact claims map to
`selected_records.json`; opcode entries retain instance index and correlation ID.
