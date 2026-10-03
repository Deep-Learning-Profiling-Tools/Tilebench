---
name: tilebench-analysis
description: Explain a TileBench backend performance gap using benchmark records, generated code, and Nsight Compute reports.
---

Source `profile/env.sh`; use `"$PY"`. Analyze the recorded winning configuration,
not an autotune sweep or regenerated benchmark. Check case, dtype, config,
implementation/capture identity, stack provenance, and GPU before comparisons.
Keep CSV benchmark latency separate from NCU timing. Missing evidence is unknown,
not zero; a mismatched or incomplete capture limits the explanation.

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
