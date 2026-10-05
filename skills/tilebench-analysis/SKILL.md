---
name: tilebench-analysis
description: Investigate TileBench kernel performance by finding the matching benchmark/configuration/profile artifacts and connecting implementation or generated code to measured bottlenecks, competing explanations, and uncertainty. Default to one backend; compare only when requested.
---

# TileBench Analysis

Find the evidence for the requested TileBench case and explain its performance,
not just its utilization. Navigation establishes the evidence; diagnosis is the
main deliverable when the user asks why a kernel is slow.
Default to one implementation, whether Triton, cuTile, or TileLang. A comparison
is optional, not a prerequisite or a reason to read other backends' code.

## Find the Case

Read [the repository map](references/tilebench-layout.md) first. Resolve the
checkout, hardware, operator, backend, dtype, shape, and default/autotuned mode
from the request and available artifacts. Ask only when an ambiguity changes the
investigation. Discover actual availability rather than requiring three DSLs.
Resolve hardware from the requested experiment and capture metadata, not the
machine running the agent; follow the map's hardware-resolution procedure.

Prefer saved evidence. Locate the benchmark row, selected configuration,
implementation or capture-time snapshot, report, and relevant provenance. Read
only the files needed for this case. Separate what the checkout currently offers
from what a particular experiment actually measured; a report filename alone
does not establish its shape, winning configuration, or source version.
If the needed artifacts are not supplied locally, follow [released artifact
discovery](references/released-artifacts.md) to find the requested saved capture
on Hugging Face. Missing local files do not imply a new profile is needed.
Use the bundled `scripts/hf_ncu_report.py` for a narrowly scoped NVIDIA download
and optional exact extraction; no handwritten downloader or new GPU run is needed.

Follow the workspace's environment instructions. This skill grants no permission
to install packages, recapture, benchmark, autotune, modify kernels, or publish.
Report missing evidence and the smallest useful next measurement instead.

## Investigate

For NVIDIA artifacts, read [saved NCU analysis](references/nvidia-ncu.md), then
the applicable [B200](references/b200.md) or [GH200](references/gh200.md) notes.
AMD and Trainium evidence can be located with the repository map, but diagnosis
on those platforms is outside this version's tested scope. Do not interpret their
artifacts with NCU counter definitions.

For a performance explanation, read [the v7-derived diagnosis workflow](references/diagnosis.md).
Start from the selected implementation's useful work, ownership, staging and
operation boundaries. Inventory the diagnostic questions, then deepen only paths
that could explain the observed behavior. Navigation-only requests need no full
diagnosis or coverage artifact.

Connect implementation/config/code -> extra work or resource/dependency effect ->
matching measurements -> plausible latency consequence. Name the mechanism and
test a serious alternative. Distinguish what creates work from what limits its
execution: fewer instructions do not imply proportionally lower latency, and low
utilization alone does not explain underfeeding or serial execution.

Use SASS/PTX selectively to resolve instruction-family, access-width, duplication,
or synchronization questions. Compiler source remains an optional route for a
material unresolved emission/capability question, not a required proof chain.

## Report

Lead with the case identity and strongest supported finding, then its code and
metric evidence, a serious alternative or contradiction, and limitations. A
single capture need not prove a bottleneck; unexplained or apparently healthy
behavior is a valid result. Keep observed facts, causal inferences, and unknowns
distinct. Do not quantify unmeasured speedups or latency contributions.

Cite code paths/lines and report/action/metric identities. Keep benchmark latency
separate from NCU duration. Save the extraction commands and exact records used
alongside the analysis in the task's output directory. If comparison is requested,
match hardware, useful work, and capture methodology; explicitly name its scope.

This combines TileBench navigation/tooling with v7-derived analysis guidance, not
a validated diagnostic system. B200 is
the initial evaluation target; GH200 notes are an untested adaptation.
