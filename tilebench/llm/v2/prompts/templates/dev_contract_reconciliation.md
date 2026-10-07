# Development-only: reconcile two independent algorithm extractions

Operator: {{operator}}
Source commit: {{source_sha}}

Below are two extractions of the same operator, produced independently from the manual Triton and cuTile implementations. Compare them stage by stage and produce three artifacts.

## Extraction from Triton

```yaml
{{triton_extraction}}
```

## Extraction from cuTile

```yaml
{{cutile_extraction}}
```

## Required outputs

1. `audit.json` (non-prompt provenance): per aspect (stages, dependencies, reduction/scan/sort structure, precision, preprocessing, mutation, intermediate storage, hardware-dispatch branches) a status of `aligned`, `mapping-only difference` or `needs-review`, each with file/function/line citations from BOTH sources; the F/Q boundary judgment; `human_reference_comparable: true|false`; `overall_status: draft|needs-review`.
2. `contract.md` (model-visible, DSL-neutral): functional semantics; inputs/outputs with aliasing and mutation rules; required logical stages and dependencies (stating explicitly which fusions, splits and reorderings are permitted); algorithm family; reduction/scan/sort constraints; precision and accumulation; preprocessing / prepacked-input boundary; permitted implementation mappings; forbidden algorithm substitutions; permitted torch operations. It must contain NO tile sizes, warp counts, stage counts, occupancy, autotune winners, latencies, or manual kernel source.
3. `evaluator_rules.json` (evaluator-only): required stages, forbidden substitution regexes with levels, required-evidence regexes, allowed torch calls, mutation/restore rule, timing boundary, tolerance source.

A genuine algorithmic divergence must be listed under `needs-review` with the exact question for a human decision; never resolve it by taking the intersection or the faster variant.
