# B200 2d_max_pooling FP32 at H=W=640

TileLang is 206.3 us, Triton 225.7 us, and cuTile 299.5 us in the autotuned CSV.
TileLang's 31.1% lower latency than cuTile is substantial; its 8.6% lower latency than Triton falls inside the study's +/-10% parity band.
The strongest supported explanation is a tradeoff between output ownership, address/predicate work, shared-memory layout conversion, and register residency.
The evidence does not establish a general DSL ranking or a compiler defect.

## Matched Work and Provenance

The unique row is line 61 of `results/B200/csv/2d_max_pooling_autotune.csv`.
Its defaults are N=4, C=128, 3x3 pooling, stride=2, padding=1, FP32 input/output.
All three implementations scan nine positions and use negative infinity outside
the input, producing 512 planes of 320x320: 52,428,800 useful outputs.
Input storage is 838,860,800 bytes and output storage 209,715,200 bytes; these are footprint calculations, not measured transactions or a traffic lower bound.
The source and winner-archive identifiers supplied for this trial are `17d2d4f6` and `9455c0bd`; they do not by themselves bind a saved profile to that source.

All report hashes match `evidence/reports.json`, dataset revision
`21037737b7e371d38f3d029367dc3967d5b31a23`. Each report has one range and one
kernel action, indexed 0/0, and identifies NVIDIA B200, capability 10.0, 148 SMs.
Triton and cuTile embedded kernel ASTs exactly match their current kernel bodies.
Triton's embedded file lacks the current autotune warmup/rep overrides; cuTile's
only file difference is the package import path. These are source correspondence
checks, not proof of benchmark/capture stack equivalence.
TileLang embeds an empty `tvm_kernels.cu` entry and a nonempty math header only.
Its grid and constant-address SASS fit the current selected implementation, but
full capture-time TileLang source/revision correspondence remains inferred.
The API reader is 2026.1.1; capture compiler/library versions, replay mode and cache-control equivalence are unresolved. All three record 44 replay passes.

## Side-by-Side Evidence

| Quantity | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| CSV latency, us | 206.3 | 225.7 | 299.5 |
| Recorded winner | 1x512, 128 threads | 1x512, 4 warps | 4x128, occupancy hint 8 |
| Captured grid / threads per CTA | 512x320x1 / 128 | 512x320x1 / 128 | 512x80x3 / 128 |
| CTA count | 163,840 | 163,840 | 122,880 |
| NCU kernel duration, us | 211.712 | 227.520 | 306.432 |
| Dynamic warp instructions, million | 62.2592 | 160.39936 | 234.94656 |
| Registers/thread; register-limited CTAs/SM | 37; 12 | 39; 12 | 64; 8 |
| Maximum / achieved active-warps percentage | 75 / 65.22 | 75 / 67.36 | 50 / 46.01 |
| ALU pipe activity / issue activity, % active cycles | 18.94 / 26.44 | 64.71 / 64.28 | 77.17 / 69.69 |
| Eligible warps/scheduler/active cycle | 0.616 | 1.609 | 1.748 |
| HBM read / write bytes, million | 839.020032 / 251.604480 | 839.145472 / 251.399680 | 922.841088 / 254.654464 |
| HBM active cycles, % elapsed | 67.16 | 62.48 | 50.09 |
| Global-load L1 sectors, million | 122.752 | 122.752 | 137.48224 |
| Global-load sectors/request | 8.065 | 8.065 | 9.247 |
| L1 sector hit rate, % | 64.60 | 64.59 | 74.49 |
| Global-store warp instructions, million | 1.6384 | 0.49152 | 0.49152 |
| Shared bank-conflict counter | 0 | 226,533 | 205,352 |

NCU gives TileLang/Triton=0.9305 and TileLang/cuTile=0.6909 for kernel duration.
CSV ratios are 0.9140 and 0.6888; CSVs remain the performance source.
Exact metric names, values, units and full report/kernel/action identities are in
`selected_records.json`; opcode instances additionally preserve correlation IDs.

## Mechanism and Tradeoffs

In `impl_tilelang.py:45`, unrolled pooling uses a register fragment and `T.copy`
to output. Captured SASS has fixed shape strides, scalar `STG.E`, and no shared
load/store path or barrier. Constant specialization also removes much of the
padded work: 27 static input-load sites versus Triton's 36, despite the same recorded 1x512 tile. Static sites describe code, not dynamic execution counts.
Triton's runtime H/W/H_out/W_out arguments leave more address/bounds machinery:
dynamic ISETP counts are 39,321,600 versus TileLang's 4,587,520, and IMAD counts
35,880,960 versus 14,417,920. This supports additional indexing/predicate work.
Triton's SASS also uses `STSM.16.M88.4`, a CTA barrier, `LDS.128`, and vector
`STG.E.128` for output layout conversion; that saves global-store instructions.
Both nonetheless have identical global-load requests and sectors, almost equal
HBM traffic, and the same register-limited residency ceiling. Extra instructions
are not extra HBM input traffic, nor do they imply a proportional latency cost.
TileLang has fewer eligible warps and substantially lower issue activity, with
exposed load-dependency signals; it cannot turn every eliminated integer
instruction into elapsed-time savings. The modest benchmark gap is consistent
with this memory/dependency floor and Triton's cheaper vector-store path.

cuTile's `ct.gather` in `impl_cutile.py:43` performs generic 3D indexing, while
its 4x128 ownership spans rows. SASS contains extensive IMAD.X/LEA/64-bit address
and bounds calculations; dynamic IMAD/LEA counts are 52,101,120/36,864,000.
It uses four shared scalar stores, a barrier, and a vector reload/store for output.
The tile has 25% fewer CTAs and 83.3% useful column slots versus 62.5% for 1x512,
but executes 46.5% more warp instructions than Triton and 3.77x TileLang's total.
Its 64-register allocation lowers resident capacity to 8 CTAs/32 warps, with
lower achieved occupancy. Local load/store counters are zero, so this is a
residency constraint, not demonstrated register spilling. High ALU activity
supports meaningful indexing execution pressure alongside memory dependencies.

cuTile's row-interleaved lane mapping also expands global-load sectors by 12%
and HBM reads by about 10%, despite a higher L1 hit rate. This supports less
efficient transaction/reuse behavior as a contributor, not a universal ideal
sectors/request threshold. Its fewer load requests do not establish less traffic.
Together these costs plausibly outweigh the selected tile's padding/CTA benefit.
An HBM-only explanation is incomplete: measured traffic rises much less than
latency, while HBM active-cycle utilization falls. Shared conversion and bank
conflicts are real, but Triton has more conflicts than cuTile and is faster;
bank conflicts alone cannot explain the ordering. Neither occupancy nor stall
ratios identify latency shares. All grids have many waves; no saved temporal
analysis here establishes a long tail, and host launch overhead is unmeasured.

The remaining uncertainty is the division of cuTile's cost among indexing,
transaction delivery, output conversion and reduced residency, plus capture
stack/cache correspondence. A matched capture manifest and a per-PC dependency timeline would discriminate those contributions; no recapture was performed.
