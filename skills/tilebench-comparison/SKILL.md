---
name: tilebench-comparison
description: Explain performance differences between TileBench implementations using matched benchmark cases, implementation structure, generated code and saved profiling evidence. Use for backend comparisons and tradeoff explanations; use tilebench-analysis for a single implementation.
---

# TileBench Comparison

Explain why the requested implementations differ, not just which counters differ.
Treat each selected configuration as a bundle of benefits and costs. A backend can
win despite lower occupancy, more synchronization, or a newer/older instruction path.
Parity and unresolved explanations are valid outcomes.

## Evidence and Scope

This skill ships alongside `../tilebench-analysis/`, which owns shared navigation,
hardware references, report readers and single-implementation diagnosis. Resolve
that sibling relative to this skill's installed location, not a hard-coded repo path.
If the shared resources are missing, disclose the incomplete installation rather
than silently substituting unrelated tooling. Install/distribute both folders.

Read the shared [repository map](../tilebench-analysis/references/tilebench-layout.md)
to resolve hardware, operator, dtype, shape, tuning mode and available backends.
Compare only the requested scope; do not assume all hardware has the same DSLs.
Use saved benchmark results and profiles. The current analysis host is not evidence
of capture hardware or software versions. Prefer capture-time source/configuration;
label current-source correspondence as inferred when provenance is incomplete.

Follow workspace environment and side-effect policies. No permission to recapture,
benchmark, autotune, install, change kernels, upload, commit or push is implied.
Missing local logs/profiles are not a reason to run GPU work: use the shared
[released-artifact discovery](../tilebench-analysis/references/released-artifacts.md)
for archived GitHub autotune winner logs and saved Hugging Face profiles. Use CSVs
for benchmark latency, not archived timing logs. Check
matching winner logs before inferring configuration from generated code; use
`../tilebench-analysis/scripts/hf_ncu_report.py` for narrowly scoped report downloads.

## Compare

Read [the comparison workflow](references/comparison.md). For NVIDIA reports use
the shared [NCU reference](../tilebench-analysis/references/nvidia-ncu.md) and the
matching B200/GH200 reference. The shared tooling does not supply AMD/Trainium
performance diagnosis; never translate NCU definitions to those artifacts.

Start with matched useful work and the benchmark gap, then investigate structural
differences capable of explaining it. Use the shared
[diagnosis workflow](../tilebench-analysis/references/diagnosis.md) to deepen a
particular implementation only where the comparative question requires it.
SASS/PTX or compiler source is conditional, not a compulsory proof chain.

## Deliver

Lead with the matched case, benchmark result and strongest supported explanation.
Include a compact side-by-side table of relevant configuration, work and counters.
Connect a code/config difference to expected and observed counter consequences,
its plausible latency effect, a serious alternative, and unresolved evidence.
Keep benchmark speedups separate from NCU timing ratios and observed facts separate
from inferred causes. Do not invent latency shares or an optimization speedup.

Retain exact per-report/action metric records, capture/download provenance, commands,
and a compact `comparison_evidence.json` following the comparison reference. Save
single-implementation coverage only when a deeper diagnosis is actually performed.
Do not read existing human analyses as the answer unless explicitly asked to assess
them; use them only after an independent investigation when evaluating correctness.
