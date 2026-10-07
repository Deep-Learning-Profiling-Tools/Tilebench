# Canonical algorithm extraction: schema and rules

Source commit (S_main): `ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`.

Two independent extractions are produced per operator, one from the manual
Triton implementation and one from the manual cuTile implementation. Each
extraction reads ONLY:

- `tilebench/benchmarks/operators/<op>/impl_torch.py` (functional semantics)
- `tilebench/benchmarks/operators/<op>/impl_<source>.py` (the ONE source)
- `tilebench/benchmarks/operators/<op>/config.yaml`
- the operator's generator in `tilebench/data/tensors.py`
- shared helpers the source imports from `tilebench/core/` (e.g. cutile_autotune)

An extraction MUST NOT read the other DSL's file, `impl_tilelang.py`,
`impl_nki.py`, autotune logs, CSV results, catalogues, or NCU reports.
Reconciliation (a separate step) compares the two extractions and reads both
sources to verify every claim.

## Output file

`docs/llm_v2/canonical/extraction/<source>/<op>.yaml`, exactly this schema
(all keys present; use `null` or `[]` when not applicable; free text in
plain English):

```yaml
operator: <op>
source: triton | cutile
source_sha: ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe
files:
  - path: tilebench/benchmarks/operators/<op>/impl_<source>.py
    functions:                       # every kernel/host function that matters
      - name: <fn>
        lines: [start, end]
        role: kernel | host-driver | helper | tuner
  - path: tilebench/benchmarks/operators/<op>/impl_torch.py
    functions: [...]
functional_semantics: >
  What is computed, in math/pseudocode terms, as defined by impl_torch.py.
  Name the reference ops used (e.g. torch.nn.functional.conv1d) and any
  reference-side preprocessing.
run_signature:
  text: "def run(...)"               # exact signature line(s) of impl_<source>.run
  positional_inputs: [names]
  keyword_params: [names]            # block_size, autotune, ... (framework knobs)
  returns: <description of returned object: single tensor / tuple / list>
  get_last_config: present | absent
inputs:
  - name: <arg>
    shape: <symbolic, e.g. "(M, N)">
    dtype: <symbolic or fixed>
    layout: <contiguous/row-major/strided/other>
    mutated_in_place: true | false
    notes: ...
outputs:
  - name: <...>
    shape: ...
    dtype: ...
    allocated_by: run | caller | reference
    aliasing: none | view-of-input | in-place-input
preprocessing:                       # host-side work before the main kernels
  - description: ...
    lines: [start, end]
    kind: cast | transpose | pack | copy | allocation | metadata | other
    inside_run: true | false         # executed inside run() (therefore timed)
    cached_across_calls: true | false
    notes: ...
stages:                              # logical stages in execution order
  - id: s1
    description: ...
    kind: kernel | host | memcpy | allocation
    lines: [start, end]
    launches: <"1" | "per tile" | "ceil(N/BLOCK)" | "k passes" ...>
    depends_on: [s0, ...]
    fusable_with: [ids] | "none" | "unclear"
    produces: <intermediate buffers>
algorithm_family: >
  e.g. "implicit-GEMM convolution", "online-softmax flash attention",
  "two-pass histogram (partial + reduce)", "bitonic sorting network",
  "LSD radix sort, 8-bit digits, 4 passes" ...
reduction_scan_sort_structure: >
  Reduction order/tree, scan direction, sort network, tie-breaking, etc.
precision:
  input_dtypes: [...]
  accumulation: <fp32 / int32 / ... per dtype>
  output_dtype: ...
  special_modes: <tf32, fp8 scaling, exact integer, fast-math ...>
  masking_or_padding_values: <-inf, 0, ...>
intermediate_storage: >
  Global scratch buffers, their shapes/dtypes, lifetime.
mutation_rules: >
  Which inputs are modified, whether run() restores them, whether repeated
  calls are idempotent.
hardware_dispatch_branches:
  - condition: <e.g. "detect_arch() == 'cdna3'">
    lines: [start, end]
    changes: ...
    algorithmic: true | false         # true = algorithm differs, false = legality/mapping only
tunable_parameters:                  # AUDIT ONLY. Never copied into a contract.
  - name: BLOCK_M
    role: tile size | warps | stages | occupancy | ...
    default_present: true | false
autotune_mechanism: >
  triton.autotune / CutileAutotuner / custom do_bench / none. Audit only.
implementation_freedoms_observed: >
  What this implementation leaves open (tile sizes, layouts, pipelining ...).
algorithm_constraints_observed: >
  What any faithful re-implementation must keep (stage structure, reduction
  order class, precision, preprocessing, scratch).
fq_boundary:
  bytes_expr: <copied from config.yaml metrics>
  flops_expr: <copied from config.yaml metrics>
  consistent_with_timed_boundary: true | false | unclear
  notes: >
    Does the formula count the preprocessing that run() performs? Does it
    assume prepacked inputs?
notes_for_review: [list of concerns, each one sentence]
```

## Rules

- Cite line ranges for every stage and branch. Do not paraphrase code you
  did not read.
- Record tunables only in `tunable_parameters` / `autotune_mechanism`.
  Never put concrete BLOCK sizes, warp counts, stage counts, occupancy or
  winner configs into any other field.
- Distinguish legality/mapping branches (e.g. smaller BLOCK_K on CDNA3,
  non-TMEM path on Hopper) from algorithmic branches.
- Say "unclear" rather than guessing.
- Do not run any code, autotuner or benchmark.
