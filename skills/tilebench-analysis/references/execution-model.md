# Reconstruct What Executes

Use this model for performance explanations, not metric lookup or artifact navigation.
Describe the selected implementation in terms of work and dependencies before
attributing a metric contrast to a language/compiler. Keep the model compact: a
table or a few annotated stages in the existing evidence record is enough.

## Operation Contract

Establish the timed operation, not just its hottest kernel. Follow the run wrapper
and benchmark invocation to locate transforms, caching, workspace initialization,
intermediate passes, vendor calls and final stores. Determine whether warmup reuses
transformed inputs and whether preprocessing is paid once, per invocation or outside
the measured region. Current harness behavior is not historical capture provenance.

Distinguish equivalent output semantics from identical algorithms. State accumulation
precision, padding, fusion and any input-layout/preprocessing contract. If an operation
has multiple stages but a report covers one, scope the model and conclusion accordingly.

## Work and Ownership Ledger

For the selected shape/configuration, derive the quantities that matter:

| Field | Establish from | Use |
|---|---|---|
| Useful outputs/operations | Shape and semantics | Common work reference |
| CTA grid and logical tile | Winner, launch and index mapping | Coverage, padding, repeated work |
| Work per lane/thread/warp | Actual layout and loop assignments | Amortization, participation, live state |
| Traversals and stages | Loop bounds and operation boundaries | Repeated input/index/transfer work |
| Data production and reuse | Loads, caches, shared/TMEM/register access | Which values are reused and at what scope |
| Completion dependencies | Producer/consumer code and emitted waits | Where ready work can be interrupted |

`outputs / launched threads` is an average ownership estimate, not proof of physical
lane layout, equal work or active participation. Account for masks, specialized roles,
redundant lanes and packed arithmetic before normalizing dynamic warp instructions.
One packed arithmetic instruction may perform multiple scalar operations; distinguish
issued warp instructions, participating thread operations and useful outputs.

Where useful, compare simple expected work to observations: loop trip count, tap count,
operand reuse scope or logical transfer payload. Explain discrepancies through padding,
predication, scalar/broadcast loads, redundant traversal or an unobserved stage. Do not
force a traffic model to equal HBM bytes under unknown cache/replay conditions.

## Live State and Dependency Sketch

When registers/residency are material, identify the plausible live objects: accumulators,
multiple outputs, prefetched inputs, address vectors, masks, conversions and epilogue
temporaries. Trace representative definitions and last uses in emitted code if available.
Ask whether unrolling lengthens live ranges or exposes independent loads/compute.

A logical accumulator count can provide a payload estimate, not the register allocator's
exact result. Fragments may distribute or replicate values; packing, allocation granularity,
temporaries and phase changes matter. Use measured resource limits to establish the
residency cost separately from the semantic benefit. Do not claim precise register
allocation causality without sufficient code/liveness evidence.

For asynchronous compute, sketch one representative iteration: issue transfer, wait for
availability, compute, signal completion, wait before reuse, advance buffers. Locate
which threads/warps perform each step and what can run independently. Configured stages
describe intended buffering; emitted dependencies determine permitted overlap, and only
runtime evidence can establish how effectively that overlap occurs.

## Compare Choices, Then Constraints

Separate three questions:

1. What implementation choice creates or removes work/state/dependencies?
2. Which execution resource or dependency makes that difference consequential?
3. Why does the net tradeoff fit the measured timing rather than only one counter?

Compare benefits and costs in the same scope. Fewer CTAs can amortize repeated work but
reduce scheduling flexibility. Vector stores can reduce issue demand but require layout
conversion. More live inputs can improve reuse/independent work but lower residency.
State which of these occurred, rather than assuming their direction from the DSL name.

Use a contradiction to choose the next investigation. If instruction counts improve but
timing barely changes, inspect instruction mix and the unchanged work/critical dependencies.
If registers rise but timing improves, investigate what the extra live state enables.
If occupancy rises without useful throughput, inspect producer/consumer roles and operand
readiness. Do not merely append the contradictory counter as a caveat.

Stop at the strongest supported implementation-level model. Mark unknown fields and
distinguish a source-supported possible overlap from a measured critical path. No per-phase
latency shares, compiler defect or hypothetical optimization gain follows from this ledger.
