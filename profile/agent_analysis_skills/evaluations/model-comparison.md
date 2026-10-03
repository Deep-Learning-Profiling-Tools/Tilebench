# Stronger Model And NCU-Style Structure

## Result

**GPT-6.1 Sol produced a substantially more specific explanation on this matmul
case with either v6 or v7.** NCU-style structure also improved Luna's evidence use:
it no longer called resource-limit counters missing and it discussed instruction
families. Sol connected the staging/completion-wait code to the observed behavior
more concretely. No agent needed compiler-source archaeology.

| Fresh trial | Quantitative audit | Explanation review |
|---|---|---|
| Sol/high + unchanged v6 | 867/867 records exact; CSV/SASS/coverage pass | Concrete lane-issued transfer versus TMA comparison, completion dependencies, resource-limit rejection and calibrated underfeeding inference. |
| Sol/high + structured v7 | 867/867 records exact; CSV/SASS/coverage pass | Similarly concrete explanation; six-lens artifact correctly records available and absent counters. |
| Luna/high + structured v7 | 867/867 records exact; CSV/SASS/coverage pass | Correct residency comparison and static opcode evidence; staging/sync hypothesis still less specifically connected to the alternative transfer paths. |

Both Sol reports are close to the requested paper-level explanation, though longer
than a final case-study paragraph. v7's benefit over v6 for Sol was auditability,
not a demonstrated new mechanism. Historical Luna/v6 indexed matmul mostly
restated low activity and overlooked occupancy evidence; the new Luna/v7 run is
better on those points, but singleton historical comparisons cannot isolate causality.

**Current preference:** Sol with the paper-style skill; use v7 when inspectable
coverage decisions are useful. v6 already worked with Sol here, so neither the
extra structure nor a custom agent runtime has been shown necessary. This is a
candidate choice, not a demonstrated general reliability claim.

## What The Better Reports Found

All selected winners use 128x128 output tiles and tensor-core instructions.
TileLang's captured SASS uses lane-issued `LDGSTS` async copies and direct stores;
the peers emit TMA load/store families. TileLang's implementation explicitly
synchronizes after tensor-memory GEMM, and SASS shows completion checking and
block synchronization before another loading region. Triton selects four stages
and K=32; TileLang selects two stages and K=64, so configuration and lowering
effects are not independently isolated.

Sol used this structure to infer operand-staging/completion dependencies that
underfeed tensor work, rather than declaring that low utilization itself was the
cause. Tensor activity is about 16% versus 27%; fraction times NCU duration is
approximately 7.9/8.1/7.9 us across the three backends. That derived quantity is
consistent with similar tensor activity spread over longer elapsed time, not a
proof of identical dynamic MMA work or of a particular stall's latency share.

The resource-only explanation against Triton is contradicted: both have three-CTA
register/shared-memory limits and 18.75% theoretical occupancy, while achieved
activity differs. cuTile has four-CTA limits and 50% theoretical occupancy. This
evidence was present in all earlier input bundles, not newly profiled for Sol.
Likewise TileLang's lower dynamic instructions than Triton reject total instruction
inflation as a common explanation against both peers. Higher DRAM traffic remains
a possible contributor, not proven redundancy or uncoalescing.

I checked the cited TileLang completion-wait/copy region and the opcode histograms;
they support code-structure observations. Stall/timeline data are still absent,
so exact overlap loss and attribution among copying, stage depth, synchronization
and epilogue remain unresolved. None of these reports proves serial execution or
an intrinsic compiler defect. PyTorch's fastest CSV result has no matching capture.

## NCU Skill Adaptation

Read the installed skill entrypoint, workflow, six analysis dimensions, diagnosis
playbook and report template. [Review notes](NCU_SKILL_REVIEW.md) explain what was
borrowed and what was not. Home skill files are unchanged.

v7 keeps v6's paper-style analysis and generic helpers, adding progressive
disclosure through [diagnostic lenses](skills/v7/tilebench-analysis/references/diagnostic-lenses.md),
a short coverage artifact, and findings-first reporting. Coverage is not six
mandatory prose sections or a requirement to invent six mechanisms. It tests
available signals before choosing a diagnosis and deepens only relevant lenses.

No copied fixed sectors/request ideals, spill-by-register-count thresholds,
mandatory full/source/PM captures, single-cause assumptions, or optimization plan.
New capture/compilation remains forbidden. Compiler source stays optional.

## Verification And Remaining Issues

All **2601/2601** submitted records match raw report values/units and action/backend
identity. Every report has one action; all nine actions are covered. CSV ratios,
fastest-reference scope and all static SASS counts independently verify. Both v7
coverage artifacts have six records and no unavailable cited counters or counters
incorrectly labelled missing. The coverage auditor has four passing tests; it
validates presence, not causal interpretation. Input/frozen hashes are unchanged.

Luna's coverage artifact says all backends have the same grid shape, contradicting
its own prose distinguishing flattened cuTile geometry from the 5x5x32 launches.
Total grid size is correctly 800. This is a residual narrative error despite
passing numerical audits. Its command log also summarizes some reads rather than
giving a complete execution trace. Sol's reports retain unknown latency shares,
capture-time package provenance and exact GPU UUID rather than fabricating them.

Luna disclosed a JSON parsing traceback under the drifted conda directory. A
parent environment probe explains this: the assigned PDF-era environment uses
that base Python/stdlib with its own package overlay. Current metadata matches
torch 2.10.0+cu130, Triton 3.6.0, cuda-tile 1.3.0, TileLang 0.1.11. This is not
evidence it used drifted packages. The failed inline parse result was not used,
per clarification. No new CUDA code was compiled; CUDA 13.2 NCU import tooling
still does not establish full native-toolchain 13.0 equivalence.

## Scope And Artifacts

Three fresh no-history agents, same previously used FP16 M=640 case and indexed
evidence, high reasoning effort. Non-skill inputs match except command worktree
paths; v6 is byte-identical to the prior skill. Historical Luna/v6 is not a fresh
fourth randomized cell. Structure plus coverage output is a combined intervention.
This is neither a repeated model benchmark nor held-out operator generalization.
Eight minutes is a soft target; logs are self-reported, shared-storage blinding is
instruction-based, and no runtime/cost ranking is claimed. No GPU profiles,
benchmark generation, commits, pushes, or home installation were performed.

Original outputs remain unchanged:

- [Sol + v6 report](results/sol-v6/output/REPORT.md)
- [Sol + v7 report](results/sol-v7/output/REPORT.md)
- [Luna + v7 report](results/luna-v7/output/REPORT.md)
- [v7 skill](skills/v7/tilebench-analysis/SKILL.md)

Audits, environment probe and final integrity checks are under `audits/`.
Archived trial metadata names the actual model; v7's reference and helper files
are included. Further confidence needs repeated tests and held-out operators,
not additional matmul-specific instructions.
