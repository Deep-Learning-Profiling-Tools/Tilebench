# TileBench Analysis Skill Candidates

Three promising skill versions from the blind profiling-analysis experiments.
These are candidates, not a claim of general reliability. Skill files, helpers,
agent reports, and archived evaluation summaries are copied unchanged.

| Version | Why retain it | Original agent reports |
|---|---|---|
| [v5](skills/v5/tilebench-analysis/SKILL.md) | Compact evidence workflow with CSV reference ranking and robust static SASS parsing. | [Luna: batched matmul](reports/v5/bmm-luna.md) |
| [v6](skills/v6/tilebench-analysis/SKILL.md) | Paper-style mechanisms; compiler-source inspection is optional, not a required detour. | [Sol: batched matmul](reports/v6/bmm-sol.md), [Luna: argmax](reports/v6/argmax-luna.md) |
| [v7](skills/v7/tilebench-analysis/SKILL.md) | v6 plus NCU-skill-inspired diagnostic lenses and explicit evidence-coverage decisions. | [Sol: batched matmul](reports/v7/bmm-sol.md), [Luna: batched matmul](reports/v7/bmm-luna.md) |

## Evaluation

The full local progression comprised 27 analysis runs across five stages, from
basic extraction through compiler pathways, paper-style explanations, and model
comparisons. This publication is a curated subset, not the entire run archive.

- [v4/v5 evaluation](evaluations/v5.md): six v4 runs and one fresh v5 run;
  9,045/9,045 submitted numeric records verified across those seven runs.
  The v5 report still overstates exclusion of tail effects.
- [v6 paper-style evaluation](evaluations/v6.md): three Luna runs;
  126/126 submitted records verified, but matmul explanations remained incomplete.
- [Sol/v6, Sol/v7, Luna/v7 comparison](evaluations/model-comparison.md):
  2,601/2,601 submitted records verified. Both Sol reports gave more concrete
  staging/synchronization explanations. v7 improved coverage auditability, not a
  demonstrated new mechanism or general reliability advantage over v6.
- [NCU skill adaptation notes](evaluations/ncu-skill-review.md): what inspired v7
  and what was deliberately not adopted.

Numeric agreement does not validate causal claims. Trials were small and were not
a randomized repeated evaluation across all operators. Compiler defects and exact
stall latency shares remain unproven. Current candidate preference is Sol with v6
or v7; v7 is useful when inspectable coverage decisions matter.

## Use

Choose one version and place its `tilebench-analysis` directory at
`.agents/skills/tilebench-analysis` in your analysis workspace. All three have the
same skill name, so do not install them simultaneously under that name. Include
the bundled `scripts/` and, for v7, `references/` directory.

Provide the agent with the selected benchmark row, winning configurations,
implementation/generated-code snapshots, and matching NCU reports. Do not supply
these example analyses as input to a blind evaluation. The skill expects the
workspace's `profile/env.sh` and compatible NCU tooling; this is not a standalone
profiler or a replacement for capture provenance.

## Archive Boundaries

Reports are the agents' original wording, including mistakes and limitations.
Their references to local `evidence/`, JSON extraction outputs, SASS listings,
commands, and historical run directories describe the original evaluation
workspace. Those evidence bundles and raw captures are **not included** here;
the report-only subset cannot independently reproduce the numerical audits.
Relative links inside archived evaluation summaries likewise retain their
original layout. Use the links above to navigate this curated package.

[PROVENANCE.tsv](PROVENANCE.tsv) maps each copied file to its original repository
path and SHA-256, allowing comparison with the full local archive. No benchmark
CSV regeneration, new GPU profiling, or edits to installed home skills were
performed for this publication.
