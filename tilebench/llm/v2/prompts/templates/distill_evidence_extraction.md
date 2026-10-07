# Source-only distillation, step 1: evidence extraction from ONE trajectory

Scope of this distillation (fixed; do not use anything outside it):
- DSL: {{dsl}} {{dsl_version}}
- Source device: {{source_device}} (trajectories from any other device are excluded)
- Training folds: {{training_folds}} (operators of the held-out fold {{held_out_fold}} are excluded)
- Condition: base only (no Enhanced trajectories)

Trajectory: {{trajectory_id}} — operator `{{operator}}`, dtype `{{dtype}}`, model `{{model}}`

Below are the rounds of this single trajectory: for each round the candidate source (or its diff from the previous round), its validity components, its measured runtime (ms) and its SOL-efficiency record. Extract observations as a JSON list. Each observation must have:

- `rule`: a conditional statement ("when X, doing Y tends to Z") about implementation choices in this DSL
- `applicability`: capability/resource assumptions it depends on (named in device-neutral terms, e.g. "shared-memory capacity per block", "tensor-core availability for dtype")
- `effect`: observed gain or failure type, with the rounds that show it
- `confounders`: other code changes that happened in the same rounds (do not claim single-factor causation when several things changed)
- `evidence`: {"supporting": [round ids], "contradicting": [round ids]}
- `limits`: what this trajectory cannot establish

Do not output a complete solution for the operator, a fixed winning configuration, or any performance claim about a device other than {{source_device}}.

## Rounds

{{rounds_block}}
