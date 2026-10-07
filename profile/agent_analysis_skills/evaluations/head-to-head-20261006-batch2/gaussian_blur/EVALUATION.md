# Gaussian Blur Paired Evaluation

## Verdict

Both skills explain the important tradeoffs correctly. The original NCU arm adds
more actionable PC-level localization and rejects a misleading rule-engine warning.
On this sample it has a detail advantage; our skill does not demonstrate a better
causal explanation. This is qualitative parent assessment, not a blinded score or
evidence of general model/skill superiority.

Original reports:
- [TileBench arm](tilebench/worktree/output/REPORT.md)
- [Original NCU arm](original_ncu/worktree/output/REPORT.md)

## Common Findings

Both recover every winner: TileLang 4x256/128 threads, Triton 8x64/8 warps,
cuTile 2x128/occupancy hint 4, at input_rows=10240, FP32, 7x7 filter.
Both correctly keep CSV times (1.1436/1.2487/1.2713 ms for TL/TR/cuTile) separate
from NCU (1.134976/1.274624/1.279904 ms). Neither treats the 8.42% TileLang
latency reduction versus Triton as a large benchmark divergence or infers statistical
significance from rounded values; the cuTile comparison is threshold-borderline.

This case is deliberately not evidence of slow TileLang: it wins slightly despite
254 registers/thread and a 12.5% theoretical warp ceiling. Both recognize that
larger per-thread work/load overlap can trade residency for reduced instruction and
transaction demand. They do not recommend higher occupancy as an automatic fix.

Both distinguish on-chip load/issue pressure from HBM bandwidth saturation:
Triton L1 throughput is 98.45% of peak, with 47.68% more global-load sectors than
TileLang, but HBM traffic is similar and DRAM throughput is only about 9% of peak.
Both identify cuTile's retained row-loop/gather bounds/address costs and its paired
FP32 arithmetic. Near-equal Triton/cuTile latency does not imply equal bottlenecks.

The TileBench arm additionally highlights cuTile's shared output conversion and
the scalar-versus-paired FMA counts. The original arm localizes Triton LG throttle
to coefficient and input load PCs, TileLang long-scoreboard samples to staged-load
dependent FFMAs, and cuTile math-throttle samples to R2UR/bounds instructions.
It also rejects the rule engine's non-fused-FP32 warning because SASS uses FFMA2.
That rule interpretation is a meaningful benefit beyond merely exporting more metrics.

Both verify Triton/cuTile kernel-body correspondence and qualify TileLang's
correspondence because its embedded CUDA is empty. Both avoid unsupported compiler
defect claims, measured latency shares or promised optimization speedups.

## Independent Audit

Sixteen non-skill files are identical between arms. All report hashes match, no
frozen inputs changed, and no checked record lacks report identity.
Each arm's 7,365 full scalar records were checked against the raw API.
Selected evidence contains 314 records for TileBench and 10,271 for original NCU,
the latter including thousands of instance records; count is not a quality score.
There are zero numeric/type mismatches in either arm. Original NCU has three
reload-dependent unit-label differences for
`derived__memory_l2_theoretical_sectors_global_excessive` (sector versus byte),
one per report, retained in the audit. That metric does not support a numerical
claim in the written report. TileBench has no unit discrepancies in this audit.

Both preserve raw SASS and opcode evidence but neither saves a standalone SASS
parser inventory for this case. Do not claim both passed an inventory-completeness
test merely because metric extraction succeeded. The numeric audit establishes
faithful extraction, not the unique cause of elapsed time.

## Limits

One sample per arm; no kernel ablations or recapture; optional common tools and
all artifacts supplied. Agent boundaries are instructed rather than security
enforced. This FP32 case cannot be substituted for the older human FP16 Gaussian
blur narrative. The skills remain frozen at the published commit throughout.
