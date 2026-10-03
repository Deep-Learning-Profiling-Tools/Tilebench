# Diagnostic Lenses

Adapted for saved comparative reports from the NCU skill's analysis-dimensions,
diagnosis-playbook and findings-first structure. Enumerate actual metric names;
examples below are discovery terms, not guaranteed availability or thresholds.
Never collect new profiles or change winners without authorization.

| Lens | Question | Available evidence to inspect | Important rejection / limit |
|---|---|---|---|
| Useful work and instruction path | Do algorithms, loop counts, primitive contracts or emitted work differ? | Selected configs and implementation; `smsp__inst_executed.sum`; dynamic families where collected; static opcode mix/operands | Static counts are not executions. A larger count need not determine latency when another resource dominates. Separate config/algorithm from compiler attribution. |
| Launch and residency | Does grid size or a limiting resource restrict concurrent work? | Grid/block/SM count, registers/shared bytes, `launch__occupancy_limit_*`, theoretical occupancy if available, achieved warps | Achieved activity is not a residency ceiling. Stalled resident warps can still count as active. Similar per-thread resources do not establish identical residency or scheduling. Resource-limit counters can support a bound even without an explicit theoretical-percent metric. |
| Memory movement and reuse | Is traffic, transaction structure, cache state or operand staging different? | DRAM/L1/L2 counters, requests/sectors, cache hits, TMA/async load and shared/local-memory paths in code | DRAM bytes are not logical input bytes. Request efficiency depends on width, lanes, alignment and useful bytes; there is no universal ideal sectors/request. Local accesses require code/context before calling them spills. |
| Issue, dependencies and synchronization | What prevents ready work from progressing? | Collected issue/stall/pipe counters, instruction dependencies and barriers, configured staging and loops | Stall ratios are not elapsed-time shares. Barrier presence is not barrier cost. High not-selected can reflect ready work; low throughput alone cannot establish a particular dependency or lack of overlap. |
| Compute pipeline | Is the relevant arithmetic path present and sufficiently fed? | Arithmetic instruction families, tensor/ALU/SFU pipe metrics, tiles and reuse | A newer instruction family does not imply faster execution. Presence is not utilization, and low activity alone does not identify why operands/issue are delayed. Non-matrix operations may not need tensor cores. |
| Balance and temporal behavior | Could work distribution, wave structure or prologue/epilogue matter? | Grid/wave geometry, per-CTA work, per-SM distributions, timeline/PM evidence if captured | Geometry indicates potential, not observed tail duration. Do not invent a timeline from averaged metrics or treat tile divisibility as proof that wave-tail effects are absent. |

## Synthesis

For each candidate mechanism, connect:
implementation/config/code difference -> extra work or dependency/resource effect
-> matching measurement -> plausible latency consequence. Check a serious
alternative: matched useful traffic, matched arithmetic, resource ceilings,
cache/replay effects, or a different dominant kernel can reject a tempting story.

Do not turn the table into an obligatory tour of every metric. Missing timeline
or stall data may limit a specific attribution while leaving an operator-level
mechanism supported. Conversely, code that could serialize does not prove the
kernel spends most of its time serialized. Compiler source is an optional route
for an unresolved lowering/capability question, not an additional required lens.

Numeric estimates from NCU rules, if available, are diagnostic estimates rather
than measured speedups; do not add them as independent latency contributions.
