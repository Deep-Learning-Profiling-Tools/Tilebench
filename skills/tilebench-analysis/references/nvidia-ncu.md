# Saved NVIDIA Report Analysis

Read only the requested implementation's report unless comparison is requested.
Resolve a compatible NCU executable and `ncu_report` Python API from the workspace;
configure its module path if needed without replacing the benchmark environment.
GPU access is not required to inspect a saved report. Current installed versions
are not proof of the software versions that generated it.

## Extract and Identify

The bundled `scripts/ncu_extract.py` inventories every range/action and exports
scalar values with exact names, units and identities. Resolve the script relative
to this installed skill; `<python>` is the workspace-approved interpreter:

```bash
<python> <skill-dir>/scripts/ncu_extract.py <report> --output <output>/ncu.json
# Optional focused export; the complete action/metric inventory is still retained:
<python> <skill-dir>/scripts/ncu_extract.py <report> --metric smsp__inst_executed.sum --output <output>/instructions.json
```

The JSON contains `actions`, `metrics`, and `errors`. Missing requested counters,
unsupported scalar values, and extraction failures are explicit errors, not zeros.
No errors does not establish capture completeness or causal accuracy. Check kernel
names, repeated launches, and expected stages against matching capture metadata.
Keep each action separate before making operation-level aggregates. Kernel duration
sums exclude uncaptured work/host overhead and can be misleading with overlap.

Enumerate metric names instead of assuming names copied from another architecture.
If only an exact export is supplied, preserve its provenance and say that the raw
report was not independently verified. CLI formatting/scaling is not exact export.

## Code and Counter Paths

Inspect the implementation and selected config to understand output ownership,
tile sizes, loop work, memory reuse, staging and synchronization. Then choose the
needed checks, rather than writing an inventory of every available counter:

- Work: `smsp__inst_executed.sum` is dynamic warp instructions. Static SASS listing
  counts are not executions or thread instruction counts; instruction families and
  operands can reveal the path but do not give its total runtime cost.
- Resources: grid/block, registers/shared memory, and collected occupancy-limit
  counters establish potential limits. Achieved active warps are not a theoretical
  residency ceiling or an eligible-warp count. High registers alone prove no spill.
- Memory: distinguish logical bytes, transactions, cache traffic and HBM traffic.
  A high L1 hit rate does not eliminate load dependencies or LSU pressure; low HBM
  throughput alone does not prove insufficient outstanding HBM requests.
- Issue: preserve the denominator of each stall metric. `per_issue_active.ratio`
  is not an elapsed-time percentage. Large stall signals need eligible/issue and
  code context before they explain latency; occupancy can change latency hiding.
- Compute: arithmetic/tensor instruction presence is not utilization. Low pipe
  activity is a symptom; investigate operand feeding, dependencies and useful work.
- Balance: launch geometry can suggest small-grid/tail effects. Averaged counters
  cannot establish the duration or shape of a tail; use temporal evidence if saved.

Read embedded SASS when it answers a specific question:

```bash
<ncu> --import <report> --page source --print-source sass
```

Record failed imports/empty listings. Missing source-line correlation is different
from missing machine code. Follow relevant compiler source only if emission or
capability remains material; no mandatory source -> PTX -> SASS -> NCU chain.

## Infer Carefully

Connect code/config -> additional work or dependency/resource effect -> supporting
measurement -> plausible performance consequence. Check a serious alternative.
Instruction count need not track latency if memory, dependencies, parallelism, or
another pipe dominates. Without a reference, diagnose constraints but do not assert
relative slowness or recoverable speedup. Rule-engine estimates are hypotheses,
not measured improvements. Missing data may leave the main cause unresolved.

Check cache-control/replay/range selection and full/targeted/reduced collection
before trusting a measurement's scope. Do not combine incompatible captures as if
they were one experiment. Keep benchmark latency and profiling duration separate.
