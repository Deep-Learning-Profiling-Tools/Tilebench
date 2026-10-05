# Diagnose the Selected Implementation

Explain performance by connecting implementation structure to measured work,
resource constraints and dependencies. Test competing explanations and preserve
uncertainty. Another backend's profile is not required.

## Establish What Runs

Read the selected config and implementation before choosing a bottleneck. Identify
the algorithm, logical outputs/useful work, tile/output ownership per CTA and warp,
loop bounds, data reuse, transfer/staging scheme, synchronization, and all operation
stages. Inspect generated code when the source does not establish the executed path.
Do not substitute current code for capture-time code without stating the uncertainty.

Inventory every captured action and relate it to those stages. Determine which
stages account for captured duration; do not extend one stage's pathology to the
whole operation. Preserve whether timing represents individual kernels, a sum,
or a benchmark measurement. Explain discrepancies rather than substituting one
timing source for another. Missing stages limit operation-level conclusions.

## Short Diagnostic Coverage Pass

These six questions organize evidence discovery, not six required findings or
official NCU sections. With one implementation, ask how it does useful work and
what could constrain it; do not silently fetch other backends as a control.

| Question | Evidence to inspect | What can reject a tempting explanation |
|---|---|---|
| Useful work and instruction path | Algorithm/loop work, selected tiles, output ownership, dynamic instructions, static instruction families/operands | Static listing length is not dynamic work. Padding, predication, lane participation and operation scope can confound count normalization. |
| Launch and residency | Grid/block/device geometry, resource allocations, limiting-resource counters, theoretical and achieved occupancy | Achieved activity is not a residency ceiling; high registers alone establish neither spills nor the latency impact of reduced residency. |
| Memory movement and reuse | Logical traffic model, measured HBM/L2/L1 traffic, transaction widths/requests, shared/local-memory and async paths | Cache hits do not eliminate dependencies; request efficiency depends on lanes/width/alignment. Local memory is not automatically spilling. |
| Issue, dependencies and synchronization | Ready/eligible and issue evidence, stall denominators, barriers/dependency chains, configured staging and emitted transfer path | A large stall ratio or visible barrier does not measure its elapsed-time cost. Low activity does not prove absent overlap. |
| Compute pipeline | Required arithmetic, emitted instruction families, pipe activity, operand staging and reuse | Instruction presence is not useful throughput; low tensor/ALU activity may result from delayed operands or insufficient work. |
| Balance and temporal behavior | All stages, grid waves, work distribution, per-SM statistics, captured timeline/PM data | Geometry suggests potential tails, not measured tail duration. Averages cannot reconstruct a timeline; divisibility alone cannot rule out tails. |

For a full diagnosis, save a compact `diagnostic_coverage.json` in the task output:
an array of `{lens,status,metrics,missing_metrics,evidence,conclusion}`. Status is
`supported`, `insufficient`, or `not_applicable`; supported means useful evidence
exists, not that a bottleneck is proven. Include report/action scope in evidence.
Metric lists contain exact inventoried names. Put unavailable types of evidence
(such as a timeline) in prose fields, not invented counter names. Inspect the
inventory before marking a metric missing. Coverage stays out of the final prose
unless it changes the conclusion; lightweight metric-only requests need no artifact.

## Choose a Mechanism and Test It

Use a candidate mechanism to decide which evidence to deepen. The following are
conditional pathways, not diagnoses triggered by a threshold:

- **Redundant work or math/index lowering:** trace loops, padding, output ownership,
  address/control work and arithmetic families. Relate dynamic counts to actual
  useful work where participation and scope are known. A count alone cannot show
  that indexing caused the count or that instructions dominate latency.
- **Resource-limited concurrency:** verify which register/shared/block limit binds
  and how it affects possible resident CTAs/warps. Then inspect achieved/eligible
  work and dependency/issue evidence. The resource ceiling is a constraint; showing
  it exists is not the same as proving it explains the observed latency.
- **HBM bandwidth versus memory dependency:** distinguish useful bytes, measured
  HBM traffic and achieved throughput from cache traffic, load instructions and
  dependency stalls. Saturation plus plausible traffic can support bandwidth limits;
  low HBM throughput plus cache hits does not establish either a bandwidth bottleneck
  or a count of outstanding HBM requests. Inspect the path feeding dependent work.
- **Transaction or shared-memory inefficiency:** relate addresses, widths, active
  lanes and alignment to collected sector/request or bank-conflict evidence. No
  universal sectors/request ideal applies to every access. Do not infer a conflict
  solely from low throughput or a shared-memory allocation.
- **Issue pressure versus exposed dependencies:** inspect load/store, arithmetic,
  control and synchronization families together with collected issue/stall signals.
  Distinguish too much work for a pipe from too little independent ready work.
  Separate extra-work origin from the resource that makes that work costly.
- **Staging or tensor underfeeding:** establish the emitted transfer/compute path
  and completion dependencies, then examine pipe activity, resource constraints and
  available stall/temporal evidence. An asynchronous opcode does not prove overlap;
  a source barrier does not prove that every phase executes serially.
- **Small-grid, tail or launch effects:** check operation granularity and hardware
  geometry, then actual distribution/timing evidence if present. Host launch costs
  cannot be inferred from an individual NCU kernel's duration.

For the strongest candidate, record: the concrete code/config feature, expected
measurement consequences, actual matching observations, and the best alternative
or contradiction. A code feature without its expected measurements is weaker;
counter patterns without code context may support only a symptom-level finding.
Missing evidence is different from evidence that contradicts the mechanism.

## Explain Without Overclaiming

Separate the **cause of work** from the **execution constraint**. Instruction count
and latency can diverge because instructions use different pipes, operands are
delayed, dependencies are hidden/exposed, or concurrency differs. Name which of
those explanations the capture supports; do not invent time shares to reconcile
the totals. An apparently healthy capture or unresolved cause is valid.

A useful operator-level inference need not identify a compiler pass. Conversely,
a compiler defect requires relevant emitted-code/source evidence, version/target
and a demonstrated selected path, not a utilization difference. Inspect compiler
source only if it can resolve a remaining material question; stop when it cannot.

Single-implementation evidence can identify constraints and suspicious work, but
does not establish relative slowness or a recoverable speedup. If comparison or
scale/dtype controls are requested, use matched existing evidence and preserve the
comparison's scope. Use `scripts/csv_ratios.py` for explicit numerator/denominator
checks; select a unique row and compare compatible units. Invalid references and
ambiguous stored-ratio formulas remain unresolved. Do not imply a universal DSL
failure from one selected configuration or one case.

Write a concise finding that connects mechanism, code and counter consequences,
then addresses a serious alternative and the remaining uncertainty. State what
smallest additional observation would discriminate unresolved candidates, without
collecting it automatically. Preserve exact metric records and rerunnable commands.
