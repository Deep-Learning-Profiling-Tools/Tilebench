---
name: tilebench-analysis
description: Produce a paper-style explanation of a TileBench backend performance gap using selected configurations, implementation structure, NCU metrics, and selective machine-code inspection.
---

Source `profile/env.sh`; use `"$PY"`. Analyze the recorded winning configuration,
not an autotune sweep or regenerated benchmark. Check case, dtype, config,
implementation/capture identity, stack provenance, and GPU before comparisons.
Keep CSV benchmark latency separate from NCU timing. Missing evidence is unknown,
not zero; a mismatched or incomplete capture limits the explanation.

Use the diagnostic lenses in [references/diagnostic-lenses.md](references/diagnostic-lenses.md)
to inventory the available evidence before choosing a mechanism. This is a short
coverage pass, not six mandatory prose sections or six diagnoses. Save decisions
in output/diagnostic_coverage.json as an array of
{lens,status,metrics,missing_metrics,evidence,conclusion}; status is supported,
insufficient, or not_applicable. metrics and missing_metrics are exact counter
names, not values inferred from names. Check the report inventory before labeling
a counter missing. Missing evidence and evidence contradicting a hypothesis differ.
Then deepen only the lenses that matter for the gap. A compact supplied index is
a discovery aid, not a diagnosis. Include instruction-family differences when code
inspection is relevant, rather than substituting static listing size for a code
comparison. Rank explanations by support; do not invent latency contribution shares.

Write findings first: benchmark gap, relevant side-by-side values, primary
mechanism with implementation/code evidence and counter support, strongest
alternative or contradiction, and limits. Keep the coverage pass in the artifact
unless it materially changes the explanation. No optimization plan is required.

Use the generic helpers beside this skill to avoid rewriting fragile extraction:

- `"$PY" .agents/skills/tilebench-analysis/scripts/csv_ratios.py <csv> --identity '<JSON row selector>' --target-column <target_latency_column> --reference-column <reference_latency_column>`.
  Repeat the reference flag for each baseline being compared, using columns in
  the same units. The helper derives numerator/denominator labels, pairwise time
  ratios, and the fastest reference among the supplied columns. State that scope;
  include every available non-target baseline when saying "fastest non-target".
  Ranking uses printed CSV values and preserves exact ties. Select a unique row.
  Optional `--stored-ratio <column>` checks formulas against printed rounding
  intervals; an ambiguous match is not a verified formula. Do not infer direction
  or reference from a column name. Explicit `--comparison` mode is also available
  for other ratios, but a free-text label does not validate its semantic meaning.
- Import saved machine code with `"$NCU" --import <report> --page source --print-source sass`,
  then run `"$PY" .agents/skills/tilebench-analysis/scripts/sass_listing.py <saved_listing>`.
  Inspect per-kernel counts and parse warnings, not just the total. Static listing
  counts include NOPs; they are not dynamic execution counts. A missing standalone
  file does not establish that SASS is absent. CUDA source-line correlation is a
  separate capability. Record failed imports and empty listings as missing evidence.

Read raw reports with `ncu_report` and enumerate ranges/actions and metric names.
A supplied `evidence/metrics.json` is an unranked exact numeric export, not a
diagnosis. Preserve raw numeric types, values, units, report path, range and action
indices. Do not substitute CLI display strings or auto-scaled units. Use
`smsp__inst_executed.sum` for warp instruction count when available and label other
counters separately. Multi-action captures need per-action identity and coverage;
do not compare one stage against another backend's entire operation. NCU duration
sums do not include uncaptured kernels or host launch overhead.

Achieved active-warp activity is not a theoretical residency limit. Cite limiting
resource counters when available. Stall counters ending in
`per_issue_active.ratio` are issue-normalized, not percentages of elapsed time.
Inspect implementation/generated code to support named mechanisms, and explain
which measurements support or contradict each. A compiler-specific cause needs
code evidence, not only counters. Distinguish observations, causal inferences,
and unresolved claims. State what would resolve the uncertainty. Link numeric
claims to report/action/metric identifiers and code claims to paths/lines. Save
rerunnable extraction commands and original helper outputs alongside the report.

Aim for an operator-level performance explanation, not compiler-pass attribution.
Start with the benchmark gap and selected implementation/configuration: determine
whether algorithms, tile reuse, loop work, execution granularity or staging differ.
Use matching NCU counters to test the mechanism and competing explanations. Inspect
SASS/PTX selectively when instruction families, transaction widths, redundant work
or synchronization structure would resolve an important question. Merely reporting
lower utilization does not explain why the kernel does less productive work.

Separate the cause of extra work from the resource or issue bottleneck through
which it affects latency. Explain why instruction-count and latency ratios may
differ using measured memory activity, resource limits, stalls or parallelism;
do not assume extra instructions always imply proportionally longer runtime.
Use matched useful work/traffic and control cases to reject alternatives when
available. Keep findings specific to the measured dtype/scale and selected winner.
No arbitrary number of metrics or complete checklist is needed in the prose.

A supported mechanism can be an inference: explain the implementation/code
difference, its measured consequences, and important contradictions or uncertainty.
Do not withhold a useful explanation solely because a compiler pass or exact
latency contribution is unknown. Conversely, counters alone cannot prove a named
compiler defect, and low activity alone cannot prove serial execution.

Consult compiler source only if a material question remains about why observed
code was emitted, a capability/constraint, or a specifically requested compiler
diagnosis. Source -> PTX -> SASS -> NCU is not a required chain. Check version and
target, follow relevant emission rules selectively, and distinguish possible
capabilities/guards from a path actually selected for this kernel. Stop once the
operator-level explanation is supported or the available evidence cannot refine it.

Before claiming evidence is unavailable, check the report's metric-name inventory
or the relevant catalogued source paths. Omitted compiler binaries do not imply
that native source text is absent. If a compact source index is supplied, use it
for discovery before searching vendored dependencies. Cite the supplied source
copy, not its original installation path. Missing kernel IR can still prevent
proving that a source branch was selected even when its implementation is present.
