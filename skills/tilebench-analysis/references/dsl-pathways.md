# Locate DSL Implementation Choices

These are investigation entry points for TileBench's selected Triton, cuTile and
TileLang implementations, not universal statements about their compilers. Follow
the actual captured path and source version; API names or defaults alone do not
establish emitted layout, resource use or scheduling. Read only the relevant pathway.

## Specialization, Addressing and Predicates

Find how shape, strides, loop bounds and layout enter the compiled kernel. Distinguish
compile-time constants from runtime scalar arguments/descriptors. Inspect decorators,
specialization parameters and wrapper construction, not just a kernel's signature.

- Triton: trace `constexpr`/specialization arguments, index tensors, masks and
  descriptor construction. A runtime-looking value may still be specialized; verify.
- cuTile: trace tensor shapes/strides, gather indices, padding/bounds behavior and
  tile constants. Do not presume a gather emits a particular descriptor/address path.
- TileLang: trace symbolic shapes and concrete factory arguments, loop unrolling,
  fragment ownership and copy regions. Fixed-shape source alone does not prove pruning.

If additional indexing is suspected, locate representative address/predicate chains
in SASS/PTX and dynamic opcode instances when available. Separate loop control, output
layout conversion and actual arithmetic from address work. Match family counts to work
and scope; an IMAD can compute an address, move a value or serve another role.

Ask whether specializing a parameter could remove work, but do not blame a compiler
for retaining a runtime quantity deliberately left runtime by the implementation.

## Ownership, Layout Conversion and Packed Work

Trace logical output coordinates through physical lanes/warps and the final store.
Check vector widths, active predicates, shared/TMEM transitions, shuffles, barriers
and store layout. Source tile dimensions do not identify physical lane assignments.

- Triton: inspect index construction, compiler-selected layouts and transitions around
  reductions/dot/store. A source store can require a layout conversion.
- cuTile: inspect tile shape/order, gather/store mappings and any role-dependent layouts.
  Count scalar-equivalent work when paired/packed arithmetic appears.
- TileLang: inspect fragment indexing, explicit layout annotations if present, GEMM
  accumulator ownership and fragment-to-output copies. Copies may require redistribution.

Explain what conversion buys: contiguous/vector transactions, reduced store issue work
or a compatible consumer layout. Establish its cost from emitted instructions and
collected traffic/synchronization evidence, not shared-memory allocation alone.

## Producer, Consumer and Completion

Follow transfer -> availability -> compute -> completion -> buffer reuse, across both
loop and epilogue. Determine whether the same participants issue transfers and compute
or whether roles are separated. Trace waits to the dependency they protect.

- Triton: inspect descriptor versus pointer loads, selected stage/warp settings, dot
  path and emitted transfer/MMA/completion code. Stage count is not proof of overlap.
- cuTile: inspect compute and transfer tile operations, compiler-managed scheduling and
  captured warp-role/register-allocation changes. Occupancy hints are not occupancy facts.
- TileLang: inspect `T.copy`, selected GEMM path, `T.Pipelined`/loop settings, pass
  options and explicit synchronization when present. Asynchronous copies can coexist
  with constraining completion waits; a source barrier does not prove fully serial work.

Wait PCs may belong to idle producers, consumers or completion roles. Relate PC samples
to those roles before describing a stall as useful-compute delay. Identify the consumer
of a delayed value, not automatically the instruction that produced it.

## When Compiler Source Helps

Consult the matching compiler version only for a remaining material question, such as
whether a chosen lowering permits asynchronous operation, why a layout transition is
required, or whether a parameter is specialized. Search the concrete API/pass/opcode
identified above. Stop if source cannot resolve the question. Unsupported capability
or defect claims require evidence of the selected path, not one backend's slower result.
