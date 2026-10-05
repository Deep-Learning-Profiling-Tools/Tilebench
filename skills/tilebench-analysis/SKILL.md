---
name: tilebench-analysis
description: Navigate TileBench implementations, benchmark cases, selected configurations, and saved profiling artifacts to investigate one backend's kernel performance. Use for TileBench evidence discovery and diagnosis; compare backends only when requested.
---

# TileBench Analysis

Find the evidence for the requested TileBench case and explain what it supports.
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

Inspect the selected implementation/configuration and relevant generated code.
Use counters to test mechanisms suggested by that evidence: work/traffic,
resource limits, issue/dependencies, compute feeding, and launch geometry.
Deepen only relevant paths; do not turn every category into a required diagnosis.
Use SASS/PTX when it resolves a material question. Consult compiler source only
for an unresolved lowering/capability question, with version/path evidence.

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

This is a first navigation-skill draft, not a validated diagnostic system. B200 is
the initial evaluation target; GH200 notes are an untested adaptation.
