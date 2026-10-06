# Explain the Difference

## Match Before Attributing

Select a unique benchmark row for the requested hardware, operator, shape, dtype
and default/autotuned mode. Establish semantics, algorithm, accumulation precision,
fusion, output ownership, padding and all operation stages. Match useful work, not
tile dimensions: different best tiles are part of the measured implementations.
Do not sweep configurations or force identical launch parameters to explain a gap
between the benchmark's selected winners.

Use each backend's recorded timing column, not an ambiguous stored ratio. Label
the numerator/denominator and units. The shared `scripts/csv_ratios.py` checks
unique-row selection and explicit ratios. Treat approximately +/-10% as weak
evidence in isolation unless the workspace or consistent scale/dtype trend provides
a stronger basis. Do not classify a backend as universally slow from one case.

For each profile, establish hardware, kernel/action coverage, selected tile/launch,
source correspondence and available capture version/cache/replay metadata. Mark
each as observed, inferred or unresolved. Different collection completeness and
missing stages can prevent an operation-level counter comparison. Benchmark CSV
latency is the performance source; profiler duration supports diagnosis separately.
If profiles disagree with the benchmark ordering, show both and qualify attribution.

Before declaring a selected configuration unknown, check kernel specialization names
against the captured signature, embedded operator source, launch geometry and emitted
loop/address constants. Recover only the parameters these establish. Traffic or
instruction models can support a configuration inference when assumptions are stated,
but ambiguous fits do not identify an observed winner. Do not substitute a source
default or a paper's configuration for the captured path. Missing manifests limit
provenance without making all existing code/configuration evidence unusable.

## Identify the Useful Contrast

Inspect each implementation's work organization before selecting diagnostic paths.
Use the shared six-question diagnosis reference as a coverage guide, not a demand
to produce six bottlenecks for each backend. Look for contrasts in:

- Algorithm/fusion and operation boundaries: comparable results can require
  different passes, launch counts, intermediate traffic or vendor paths.
- Ownership and loop structure: tile size, output work per CTA/thread, padding,
  repeated traversal and amortization of fixed per-tile work.
- Instruction work: indexing, math, conversion, control and transfers. Match
  action scope and distinguish static opcode sites from dynamic warp executions.
- State placement and reuse: registers, shared memory, TMEM, cache and staging;
  logical traffic models versus actual HBM/cache/TMA traffic.
- Resource and dependency tradeoffs: residency limits, ready/eligible work,
  operand delivery, synchronization and evidence of overlap or lack of it.
- Distribution and timing: grids, waves, variable work, per-SM statistics and
  actual saved temporal evidence. Averages cannot measure a temporal tail.

Prefer a few differences that explain the result over a table of every metric.
Normalize only when work and participation are understood. Sectors/request depends
on access width, active lanes and broadcast versus contiguous patterns. Traffic
through different cache/transfer paths is not automatically redundant HBM work.
Read inventories rather than treating an uncollected metric as zero.

## Build a Tradeoff Explanation

For a candidate mechanism write:

1. The concrete source/configuration or emitted-code difference.
2. How it changes useful or overhead work, reuse, resources or dependencies.
3. Its expected counter consequences and the actual matching observations.
4. Why that difference could affect the observed benchmark ordering or gap.
5. The strongest contradiction/alternative and what remains unmeasured.

Explain both sides of an important tradeoff. A larger tile may reduce repeated
work while reducing residency; a staged path may add instructions while feeding
compute more effectively. Evaluate the net benefit from the matched benchmark and
supporting evidence, not from one unfavorable counter. Low occupancy, many barriers,
high registers or low pipe activity alone are not a relative-performance diagnosis.

When instructions and latency do not scale together, inspect instruction mix,
dependency hiding, pipe pressure and memory/staging costs. Do not force a linear
instruction-to-latency relationship or infer waste from stall sample fractions.
In asynchronous/role-specialized code, waits can belong to idle participants while
others do useful work. Temporal or critical-path evidence is needed for time shares.

Useful conclusions can stop at an implementation-level tradeoff. Deep compiler
inspection is justified only when an unresolved emission/capability question could
change the explanation. Version/target and the actual path matter for a compiler
defect claim; different utilization is not enough.

## Compact Evidence Record

Save `comparison_evidence.json` as an object with:

- `case`: hardware, operator, dtype, shape, semantics and tuning mode.
- `implementations`: backend, benchmark row/path/value/unit, selected configuration,
  profile/action identities, source path and provenance status/limits for each.
- `timing`: explicitly named benchmark ratios and separate profiler durations/ratios.
- `mechanisms`: an array of `{claim,status,code_difference,expected_consequences,
  observations,alternative,limits}`, where status is observed/inferred/unresolved.

Observations cite exact report/action/metric identities and code paths. Keep numeric
records in an exact extraction rather than rounding the only copy of the evidence.
This record makes the explanation reviewable; it is not a numerical confidence score
and does not establish causality by itself. Metric-only requests need no full causal
record. Report the smallest useful additional observation without collecting it.
