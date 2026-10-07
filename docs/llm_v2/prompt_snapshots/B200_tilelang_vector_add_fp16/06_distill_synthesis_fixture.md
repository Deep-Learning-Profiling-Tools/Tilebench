# Source-only distillation, step 2: synthesize an Optimization Skill

Scope (fixed): DSL tilelang 0.1.11; source device B200; training folds B, C; held-out fold A excluded; base-condition trajectories only; generation models pooled: gpt, claude.

Input: the per-trajectory observations below, extracted under the same scope from 2 trajectories (t-synthetic-1, t-synthetic-2).

Write an Optimization Skill for tilelang as Markdown. Requirements:

1. Every rule is conditional and states its applicability (capability and resource assumptions in device-neutral terms), the observed benefit or failure mode, and its limits.
2. Every rule cites supporting and contradicting trajectory IDs from the input. Rules with no supporting evidence are not allowed.
3. When several changes co-occurred, describe the association; do not assert single-factor causation.
4. Do not include operator-specific complete solutions, fixed winner configurations, numeric tile recipes for named operators, manual-kernel tuning values, or performance conclusions about devices other than B200.
5. Do not mention the held-out fold's operators.
6. Organize as: scope statement; general implementation guidance; data-movement patterns; compute-mapping patterns; pipelining/scheduling; numerical-robustness pitfalls; common failure modes and how they were resolved; limits of this evidence.

## Observations

### t-synthetic-1 (layernorm)
[synthetic observation JSON]

### t-synthetic-2 (softmax)
[synthetic observation JSON]
