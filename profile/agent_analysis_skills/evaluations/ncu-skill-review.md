# NCU Skill Structure Review

Read the installed `ncu-report-skill` entrypoint and references for workflow,
analysis dimensions, diagnosis playbook and report template. Home skill files
are unchanged. This experiment uses the study's existing captures and constraints,
not the installed skill's collection defaults.

## Useful Structure

- Short entrypoint routes to detailed references rather than embedding every case.
- Analysis lenses cover launch/resources, imbalance, stalls, compute pipeline,
  temporal behavior and memory; each states a question and relevant evidence.
- Pattern guidance separates signals, possible cause and exceptions.
- Findings-first report links headline measurements to mechanisms and caveats.
- Structured numeric extraction and rerunnable artifacts keep measurements inspectable.

v7 borrows those organizational decisions. It adds a short coverage pass with
explicit available/missing metric names, then deepens relevant lenses only. The
coverage artifact is not an instruction to invent diagnoses or write six prose
sections. The comparison remains benchmark-first and implementation-aware.

## Adaptations For This Study

- Do not require standalone CUDA harnesses or full/source/PM recaptures for every
  saved-report analysis. User scope forbids new compilation/profiling here.
- Do not assume a single dominant cause can always be identified quickly. Multiple
  implementation/config effects and incomplete targeted captures can be confounded.
- Do not copy fixed sectors/request ideals. Instruction width, active lanes,
  alignment and useful bytes determine the transaction interpretation.
- High register usage is a resource signal, not proof of spills. Local-memory
  operations need code/context; explicit per-thread arrays also use local memory.
- Achieved active-warps metrics are not counts of ready/issuing warps. Resident
  stalled warps can remain active; activity gaps need geometry/temporal context.
- Long-scoreboard and low DRAM activity do not locate latency specifically in L1.
  Interpret dependency and cache evidence together, without inventing sampled PCs.
- Treat stall ratios as issue-normalized, not fractions of elapsed time. Preserve
  units and counter definitions; do not impose heuristic bounds from prose.
- A potential incomplete wave or low tensor activity is not a measured tail or
  proof of absent overlap. Missing timeline/stall data limits that attribution.
- NCU rule estimates, if present, are estimates, not measured counterfactual gains
  or independently additive latency shares. No optimization plan is required.

These caveats are general domain safeguards, not matmul-specific diagnosis hints.
The new reference contains no expected result for an operator. The experiment
tests whether this structure improves evidence use without deeper compiler tracing.
