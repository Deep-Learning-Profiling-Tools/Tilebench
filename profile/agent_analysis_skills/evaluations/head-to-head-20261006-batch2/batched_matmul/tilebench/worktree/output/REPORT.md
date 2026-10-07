# B200 Batched Matmul, FP16, Maximum Case

For BATCH=32 and M=N=K=640, the autotuned CSV records TileLang at 51.4 us,
Triton at 29.3 us and cuTile at 31.8 us. TileLang is 1.754x slower than Triton
and 1.616x slower than cuTile. Triton/cuTile differ by 8.5%, within the 10%
parity convention. The strongest supported explanation for TileLang's gap is
less effective operand delivery and completion-dependency hiding around tensor
compute, rather than missing tensor cores or simply more executed instructions.
This is an implementation-level inference with capture limits below.

## Matched Work and Selected Winners

The operation produces FP16 C[b]=A[b]@B[b] with FP32 accumulation. Useful work is
16,777,216,000 FLOPs, with no shape padding. Each operand/output contains
13,107,200 elements across the batch; unique A+B+C bytes total 78,643,200.
Each backend owns one 128x128 output tile per CTA: 5x5x32 = 800 output tiles.
Their selected K tile differs; these are each backend's logged winners, not a sweep.

| Quantity | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| CSV latency, us | 51.4 | 29.3 | 31.8 |
| Selected M/N/K tile | 128/128/64 | 128/128/32 | 128/128/64 |
| Selected pipeline/group | 2 stages / 1 | 4 stages / 1 | occupancy hint 4 / group 1 |
| Captured block threads | 128 | 128 | 256 |
| Captured registers/thread | 137 | 135 | 64 |
| Captured shared memory, bytes/block | 67,584 | 66,608 | 50,452 |
| Register/shared CTA limits, each | 3 / 3 | 3 / 3 | 4 / 4 |
| Achieved active warps, % of active-cycle peak | 6.123 | 15.314 | 27.889 |
| Tensor active cycles, % of elapsed-cycle peak | 16.128 | 24.633 | 22.134 |
| Dynamic warp instructions | 4,020,000 | 5,448,810 | 1,538,634 |
| NCU kernel duration, us | 48.576 | 32.064 | 34.848 |
| NCU HBM reads / writes, bytes | 29,588,224 / 3,688,192 | 52,454,656 / 51,616,000 | 52,470,784 / 50,128,640 |

CSV source: `results/B200/csv/batched_matmul_autotune.csv`, unique line 59; winner
source: `evidence/batched_matmul_autotune.json`. The independent TileLang
winner log agrees. All NCU numbers refer to range 0/action 0 of the backend's
`evidence/<backend>_fp16.ncu-rep`. Exact names, values, units and action identities
are preserved in `selected_records.json`; rounded percentages above are display only.
Torch is the fastest non-TileLang backend at 23.3 us, making TileLang 2.206x slower;
no Torch profile is supplied, so no mechanism is assigned to that reference gap.

## What the Captures Actually Establish

All report hashes match `evidence/reports.json`, dataset revision
`21037737b7e371d38f3d029367dc3967d5b31a23`. All identify NVIDIA B200, capability
10.0, and one BMM action. The supplied source/archive identifiers are `17d2d4f6`
and `9455c0bd`; these are input provenance, not independently proven capture revisions.
Embedded Triton and cuTile BMM function ASTs equal the current functions, including
their decorators. Wrapper/import differences exist; this proves kernel-body
correspondence, not the complete source revision or installed compiler versions.
cuTile's specialization suffix `I10_I128_I128_I64_I5_I5_I1` explicitly matches
10 K tiles, the selected tile sizes, 5x5 output grid and group 1.
Triton's geometry/resources and TMA loop fit the winner; its name lacks full config.
TileLang embeds template headers but its `tvm_kernels.cu` entry has empty content.
Its 5x5x32 launch, double-buffered shared copy addresses, 640-related strides and
tensor path fit the logged winner; exact capture/config correspondence remains inferred.
No capture command/version manifest is supplied. TileLang records 9 replay passes
and backed-up memory; peers record 44 passes without that backup metric.
TileLang lacks eligible-warp, issue, stall and bank-conflict counters. Current harness
source describes entry-cold profiling after warmup with intra-operator reuse; it
does not prove the historical captures used identical settings.

## Causal Contrast

TileLang's current `impl_tilelang.py` disables warp specialization, uses two stages,
copies A/B into shared memory, and explicitly synchronizes after each TMEM GEMM.
Saved SASS confirms `LDGSTS.E.BYPASS.128` lane-issued copies, `UTCHMMA`, `UTCBAR`,
phase-check completion waits and block barriers; see `tilelang_sass.txt` around
PCs `0x7fbdf35bc300` through `0x7fbdf35bc830`. There is real copy pipelining,
so this is not evidence that all memory and compute execute serially.
The wait-to-next-work dependencies and distributed address/copy work can expose
operand-delivery latency. Lower achieved warp and tensor activity support that
inference despite the same three-CTA resource ceilings as Triton.
Saved Triton/cuTile SASS instead shows `UTMALDG.3D` and `UTMASTG.3D`, together
with Blackwell `UTCHMMA` and TMEM loads. Triton's source pretransposes and caches B;
the warmed repeated-input operation reuses that transpose, not a new transpose kernel.
This layout/preprocessing contract limits generalization.

The tradeoff is visible: Triton executes more warp instructions than TileLang yet
finishes sooner and keeps tensor compute busier. Instruction total cannot explain
the ordering; memory-transfer path, overlap and dependency exposure matter.
cuTile has fewer instructions and higher active occupancy than Triton but similar
benchmark latency. Its role-oriented SASS includes register deallocation and waits.
Eligible warps/scheduler are 0.05262 versus Triton's 0.21804; long-scoreboard
`per_issue_active.ratio` values are 82.654 versus 8.435 (API unit `inst`). These
aggregate-participant ratios are not elapsed-time shares or proof that useful compute
stalls ten times longer; they do not establish a cause for the parity-sized gap.

## Alternatives and Limits

Occupancy collapse from a smaller residency ceiling is insufficient: TileLang and
Triton have matching limiting CTA counts. Both see 1.802 scheduling waves/SM;
cuTile sees 1.351. Such small wave counts allow scheduling/tail effects, but saved
averages provide no tail duration. TileLang bank conflicts, spills and host launch
cost are unestablished. All backends use tensor instructions; absence is rejected.
TileLang HBM writes are far below logical output size, while peers' writes exceed it;
cache/writeback and replay differences prevent treating these bytes as comparable
useful traffic or assigning the gap to HBM bandwidth. This is the strongest capture
alternative to the operand-feeding inference, not evidence of omitted output work.
NCU TileLang/Triton and TileLang/cuTile ratios are 1.515x and 1.394x, smaller than
the CSV ratios but with the same ordering; use CSVs for benchmark claims.
The smallest discriminating additional evidence would be the TileLang capture
manifest/generated kernel plus matched dependency/eligible-warp localization.
No new collection was performed, and no compiler defect or optimization speedup is claimed.
