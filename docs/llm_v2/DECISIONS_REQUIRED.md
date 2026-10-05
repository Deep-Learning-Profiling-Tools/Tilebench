# DECISIONS_REQUIRED — study-owner decisions before any live v2 campaign

Nothing below was decided by the framework; each item names the artifact
that encodes the decision and the file to change. Items marked
**[protocol]** touch the v2 method text; the code stays on the v2 reading
until the owner rules.

## A. Contracts (per operator; details in `docs/llm_v2/CANONICAL_AUDIT.md`)

Every contract is `draft` (37) or `needs-review` (8); none is `approved`.
Approval is recorded by setting `status: approved`, `approved_by`,
`approved_on` in `tilebench/llm/v2/contracts/data/<op>/audit.json` (the
loader refuses live use otherwise).

The eight `needs-review` operators and their blocking question:

| operator | needs-review aspects | question |
|---|---|---|
| batched_matmul | preprocessing, intermediate_storage | untimed cached B^T repack on one side (A1) |
| matmul_fp32_fp16_fp8 | preprocessing, intermediate_storage | same (A1) + plain-dict `_bt_cache` stale-address hazard + fp32 reference precision (allow_tf32 interaction) |
| matmul_int8 | preprocessing, intermediate_storage | same (A1) + arbitrary K_b support + process-wide `allow_tf32=False` side effect |
| streamk_matmul | preprocessing, intermediate_storage | same (A1) + uncounted zero-fill/scratch + fp32-atomic non-determinism accepted? |
| destindex | preprocessing, mutation | output-buffer policy: fresh outputs per call (reference semantics, 2x timed traffic) vs persistent identity-keyed buffers (manual boundary) (A6) |
| gaussian_blur | precision | per-tap product in fp32 or in the input dtype (A7) |
| quantize_global | — | operator is a plain fp32→fp16 cast at S_main (A4) |
| radix_sort | — | contract leaves digit width / pass count free while bytes_expr hard-codes 16 passes (A2) |

Draft contracts with non-blocking review items (metric conventions,
unexercised tie/NaN rules, square-grid assumptions, bytes conventions
differing between layernorm 3n and l2_norm 2n, bitonic fusion freedom) are
listed per operator in `CANONICAL_AUDIT.md`.

Cross-cutting questions raised by the two independent extractions:

A1. **Prepacked / cached operands (F-Q boundary).** The manual Triton
`batched_matmul`, `matmul_fp32_fp16_fp8`, `matmul_int8` and `streamk_matmul`
transpose B once inside `run()` and keep it in a per-tensor cache, so their
timed steady state consumes a K-major B the formula does not charge; the
manual cuTile versions consume B as given. Decide per operator: (i) the
contract requires the generated `run()` to accept B in its given layout and
pay any transpose inside the timed call (then the manual Triton reference
is `human_reference_not_comparable`), or (ii) an explicit prepacked-input
contract is declared (B^T supplied by the generator) with `bytes_expr`
re-audited. The contracts are written under (i) with the item listed as
open; the audit marks `human_reference_comparable: false`.

A2. **Uncounted in-run traffic in `bytes_expr`.** `rope` (clone),
`streamk_matmul` (zero-fill, fp32 scratch round trip), `histogramming`
(scratch zero-fill + atomics), `batch_normalization` (scratch), `bitonic_sort`
(O(log² M) passes), `top_k_selection` (per-level candidates). Decide whether
T_SOL keeps the frozen compulsory-traffic formulas (v2 text: "F/Q from the
existing formulas") and the discrepancy is only documented. **[protocol]**

A3. **`flash_attention` flops_expr is dense while the task is causal**
(about 2x overstated); `block_sparse_attention` flops_expr hard-codes the
generator's block window. Keep as frozen (v2) or correct before freezing
T_SOL? **[protocol]**

A4. **`quantize_global` is a plain fp32→fp16 cast at S_main** (no absmax,
no scale). The contract describes exactly that; confirm the operator keeps
its name and semantics for the study.

A5. **`histogramming`** at S_main uses global-atomic private rows + column
reduce on both sides (PR #317's `tl.histogram` version is not merged).
Confirm S_main semantics are the study's; a later merge would require a new
contract revision.

A6. **Output aliasing**: `destindex` returns persistent clones cached per
input identity in both manual versions; `bitonic_sort`/`top_k_selection`
return views of in-run scratch. The contracts require a fresh (or
run-allocated) output per call and forbid cross-call caches; confirm.

A7. Precision details flagged mapping-only vs needs-review by the audit
(e.g. `dropout` divide-by-(1-p) vs multiply-by-scale; `flash_decode`
epsilon absent in kernels; `gaussian_blur` per-tap product dtype). Rule on
each in CANONICAL_AUDIT.

## B. SOL model

B1. **Arithmetic mode per task** — `tilebench/llm/v2/manifests/arithmetic_modes.yaml`
is `status: proposed`: fp32 GEMM/conv/attention-class operators map to
`tf32` (the manual kernels use TF32), other fp32 operators to `fp32_vector`,
integer operators to `int_vector` with audit flag `work_metric_not_flops`,
copy/transpose/scatter operators flagged `memory_bound_candidate`. Approve,
or change entries. T_SOL is `peak_missing` for every task whose mode has
no peak value (see B2) — no efficiency is computed for them until resolved.

B2. **Peak table**. `tilebench/data/peak_performance/B200.json` has
tensor-core values only (its `fp32` entry is explicitly TF32; measured
`peak_bw_GBs` 6539.4). No `fp32_vector`, `int_vector`, no GH200/MI300X/Trn2
file exists. The device-context audit found the live NVIDIA HGX page listing
dense FP8/FP16/TF32 at 2x B200.json's values and FP32 (vector) 600 TFLOPS
per 8 GPUs (75 TFLOPS/GPU). Decide: which source is authoritative for
B200; whether to add `fp32_vector`/`int_vector` entries; and provide
GH200/MI300X/Trn2 peak files (measured bandwidth like B200's, or datasheet)
before cross-device E(B) is computed. **[protocol: "P_peak for the declared
arithmetic mode" needs these numbers]**

B3. **`memory_only` mode** (T_SOL = Q/BW) exists in the schema but is not
assigned to any task; assigning it requires approval per task.

## C. Study configuration

C1. **Model ids and decoding settings** — `manifests/models.yaml`: generator
`gpt`/`claude`, distiller, (optional) reviewer. All `null`, status `unset`.
The legacy defaults (`gpt-5.5`, `claude-opus-4-7`, effort `xhigh`,
`max_tokens 128000`) are NOT inherited.

C2. **Fold assignment** — `manifests/folds.yaml` is `proposed` (families
kept whole; 15/15/15). Set `status: frozen` + `approved_by` before any
distillation.

C3. **Context limits** (`study.yaml:context_limits`) — reference ≤120k
chars, device ≤30k, contract ≤20k, optimization ≤40k, functional reference
≤20k, total ≤260k. Current drafts: Triton 33.8k, cuTile 33.0k, TileLang
24.2k, devices 7.9–9.5k. Confirm the caps (they are the "pre-declared common
rule" for over-length documents).

C4. **Feedback diagnostic rule** — 60 lines / 6000 chars with
`[line withheld]` scrubbing of scoring terms. Confirm.

C5. **Transport retries** — 3 per attempt, logged separately; after that
the trajectory is `incomplete` (resumable). Confirm.

## D. Assets and permissions

D1. **Repository licence**: no LICENSE file exists. Shareability of the
public Reference Skills (derived from `skills/*-guide/SKILL.md`) and of the
review bundle depends on it.

D2. **TileLang Reference** is derived from `~/.claude/skills/tilelang-guide/SKILL.md`,
a local unlicensed asset, plus package introspection: registered as
`permission: internal`. Confirm it may be sent to OpenAI/Anthropic.

D3. **NKI**: `~/.claude/skills/nki-guide/SKILL.md` is private; no Reference
Skill was derived; `reference/nki@beta5` is metadata-only. Confirm licence,
API-sendability, and whether generated NKI kernels / distilled skills may
leave the private repository (`docs/llm_v2/NKI_HANDOFF.md §4`).

D4. **Approval of drafts**: Reference (3) and Device (4) skills are `draft`;
the loader injects only `approved`/`frozen` in live mode. Device snapshots
for GH200/MI300X/Trn2 are `verified_on_device: false` until the capture
checklist is run there.

D5. **B200 peak inconsistency** (see B2) also appears in the B200 Device
Context text (both figures quoted, [S2] kept authoritative).

## E. Protocol items the framework implements by a specific reading (confirm)

E1. The generated `run()` is called positionally with the operator's
generator outputs, exactly like `impl_torch.run`; no `block_size`/`autotune`
kwargs are passed (stated in the system prompt). Many draft contracts still
describe the manual call form with those framework keywords ("may be
accepted and ignored"); `top_k_selection`'s draft says an implementation
"may honour `block_size`". Decide whether to strip these mentions from the
contracts before approval (they are not tunables, but they are legacy
interface noise). **[protocol: "核对 run() 的真实签名"]**

E2. Graph timing: a failed CUDA-graph capture is recorded and the launch
runs eagerly (`timing_execution_mode: eager`, `capture_succeeded: false`);
such a candidate is still valid. If the owner prefers "capture failure =
timing_error", change `evaluation/timing.py`. **[protocol]**

E3. Unknown usage: a round with unknown cost makes E(B) undetermined only
for budgets above the last known cumulative cost (every later attempt costs
≥1 token). **[protocol wording: "标曲线不可精确计算的范围"]**

E4. `review_required` verdicts (static evidence of level `suspicious`,
e.g. module-level caches, `os`/`sys` imports, timing primitives) block the
trajectory until `resolve_review`; the optional LLM reviewer is disabled.

E5. Trn2 eligibility is `needs_review` for all 110 tasks until registered
on the device; MI300X `fp8_e4m3fn` is `unsupported` from the manual
campaign's evidence.
