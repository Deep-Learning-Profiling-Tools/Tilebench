# Matched Skill Trial

## Question

Does TileBench's comparison skill produce a more useful, reliable explanation
than the installed original NCU report skill, given identical accessible evidence?
This tests diagnosis with supplied artifacts, not artifact discovery or collection.

## Controls

- Model: GPT-6.1 Sol, high reasoning, fresh context for each arm, sequential runs.
- Case: B200, 2d max pooling, FP32, maximum H=W=640; all three DSLs.
- Repository: direct-runtime commit `17d2d4f6e1d4c3e3c5a015b3c8b5f45f14b9e95b`.
- Winner logs: archive commit `9455c0bd76a0b09febee3080ce95f438f600896f`.
- NCU artifacts: dataset revision `21037737b7e371d38f3d029367dc3967d5b31a23`.
- Same raw reports, logs, source, CSV, interpreter, optional extraction tools,
  output contract and side-effect restrictions. No human/previous-agent narratives.
- Only assigned skill instructions and their companion references differ.
- Original skill is the installed local snapshot, unmodified, not a claim about
  the latest upstream revision. Its README links MIT Han Lab's KDA project.
- Saved evidence only: collection instructions cannot authorize GPU work.
- Isolation is instructed, not a security-enforced filesystem boundary.

## Evaluation Criteria (Declared Before Results)

1. Recover the selected case and all three winning configurations correctly.
2. Use CSV benchmark latencies separately from NCU durations; correct ratio direction.
3. Preserve exact NCU values, units, action identity and instruction-count metric.
4. Explain differences with implementation/configuration evidence and compatible
   counters, rather than ranking counters alone.
5. Address why instruction-count differences need not predict proportional timing.
6. Distinguish observations from causal inference and unsupported optimization claims.
7. Handle missing metrics and incomplete capture/source provenance honestly.
8. If used, parse SASS completely and distinguish static from dynamic instruction counts.

Parent evaluation will preserve original reports and audit raw values independently.
No changes to the skills will be made during this pair. One case and one sample per
arm cannot establish general superiority; order is not randomized and extra helpers
make this different from a completely stock end-to-end NCU-skill deployment.
