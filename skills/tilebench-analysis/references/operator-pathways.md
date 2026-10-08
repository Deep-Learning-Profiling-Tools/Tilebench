# Deepen the Relevant Operator Structure

Use only the family relevant to the selected implementation and unresolved contrast.
Operator names are routing hints, not proof of an algorithm: inspect source first.
These pathways supply questions and discriminating evidence, never expected DSL winners.

## Matmul and Attention

For GEMM-like work, identify output tile, reduction tile/trip count, accumulator
precision and mapping, operand layout and reuse, transfer payload per iteration,
and epilogue. Compare the selected configurations' repeated work, not just tile area.
For attention, additionally identify query/key/value ownership, sequence masks,
softmax/rescaling reductions, intermediate lifetime and the attention-loop boundaries.
Separate the matrix products from scalar/reduction work that feeds or joins them.

If tensor activity is low, reconstruct a representative producer/consumer iteration:

1. Which input-layout transform or descriptor is prepared, and when is its cost paid?
2. How are operands delivered and made available to compute participants?
3. What compute issues before completion, and what must wait before the next issue?
4. Which buffer/accumulator can be reused, and what makes reuse legal?
5. How is the final accumulator drained, converted and stored?

Inspect prologue, steady state and drain separately when code supports that distinction.
Different reduction tile sizes can change both issue granularity and wait frequency.
Locate staging buffers and selected stage counts, but infer effective overlap only from
dependency structure plus compatible measurements. A pipeline label is insufficient.

Compare register/shared/TMEM footprints to each backend's actual resource ceiling;
an accumulator layout or double buffering may have benefits despite higher state.
When configurations have equal residency ceilings but different achieved activity,
resource allocation alone cannot explain the difference. Investigate feeding, roles,
phase changes, work distribution and capture comparability.

Check whether pretransposed/cached inputs, fused epilogues or auxiliary kernels change
the timed contract. Unprofiled preprocessing cannot be assigned zero latency. Do not
sum overlapping kernel durations into a serial operation model without evidence.

## Stencils, Pooling and Direct Convolution

Reconstruct each output's neighborhood and how outputs are assigned to lanes/threads.
For direct forms, derive tap/loop work and boundary handling; distinguish a reduction
such as max from multiply-accumulate arithmetic. Convolution may instead use transformed
or GEMM-like work; route to that execution path rather than assuming direct taps.

When instruction work differs, inspect constant-folded bounds, retained loops,
predicates, address strides and arithmetic packing. Use dynamic opcode counts to
distinguish indexing, accumulation, coefficient access and layout conversion.
Static tap sites do not establish executed tap work under predicates and reuse.

When register use differs, trace the output bundle and loaded neighborhood values:
are values reused across outputs, loaded early for independent work, or duplicated?
Identify representative lifetimes instead of assuming every unrolled load stays live.
Compare the reuse/independence benefit with the measured residency/eligible-work cost.

When cache traffic differs, inspect representative input and coefficient accesses
separately. Reconstruct lane strides, widths, overlap and broadcast behavior for interior
and edge tiles as relevant. Correlate requests/sectors with useful work, L1/LSU pressure
and load-PC samples. High hits and low HBM throughput do not remove on-chip issue cost.

Trace output conversion/store separately: vectorization can reduce issue work while
requiring shared redistribution or synchronization. A backend may improve arithmetic
or tile utilization yet lose on bounds, transactions or conversion costs.

## Reductions, Scans and Elementwise/Fused Operators

Establish reduction length/axis, elements per participant, local versus cross-warp
combination, output count and any multi-pass/global combination. Trace intermediate
traffic and the final reduction/store. Padding and inactive lanes can dominate small
or awkward shapes even when launch threads appear equal.

For instruction contrasts, separate arithmetic lowering, conversions, approximate
math, masks/indexing and reduction communication. Verify precision/semantics before
calling fewer instructions an efficiency improvement. Fusion may save launches and
traffic while creating live-state or scheduling costs; compare the complete timed path.

For small work, distinguish host/dispatch/graph timing from device kernel time. Without
operation-level timing evidence, an NCU kernel does not establish launch overhead.

## Resolve a Contradiction

If a predicted cost is present but timing does not track it, select the relevant
work/dependency contrast above. Determine whether it is off the limiting path, buys
reuse/parallelism, is hidden by other work, or is confounded by capture boundaries.
State which possibilities the evidence distinguishes. A deeper analysis may conclude
that two explanations remain compatible; do not convert a plausible execution model
into a measured critical-path or latency-share claim.
