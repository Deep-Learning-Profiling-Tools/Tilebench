# VALIDATION_2026-10-05 — real ten-round acceptance of the v2 execution chain on B200

Campaign `validation_b200_2026-10-05` (`outputs/llm_v2/validation_b200_2026-10-05/`
run cache; publication copy under `artifacts/llm_v2/validation_b200_2026-10-05/`).
Run type `validation`: engineering acceptance of the shared execution chain
with real models and the real device. Unscored, never a distillation
source, never mixed into formal curves (`metrics` labels it
`scoring_note: validation run`). Nothing below is a formal Base result.

## 1. Setup

| Item | Value |
|---|---|
| Host / device | dgx003, NVIDIA B200 (driver 595.58.03), arch `blackwell`; cuda-tile 1.5.0, triton 3.6.0, torch 2.10.0+cu130 |
| Condition | Base (Reference Skill + Device Context + contract); no Optimization Skill |
| Context assets | draft Triton 3.6.0 / cuTile 1.5.0 references and the draft B200 device snapshot (public, granted to both providers; hashes in every `trajectory.json.content_hashes`); draft contracts (37 draft / 8 needs-review, none approved) |
| Models | `gpt` = `gpt-6.1-sol`, Responses API streaming, `reasoning.effort=xhigh`, `max_output_tokens=128000`; `claude` = `claude-opus-5-5`, Messages API streaming, `output_config.effort=xhigh`, `thinking={type: adaptive}`, `max_tokens=128000`. Keys from `OPENAI_API_KEY` / `CLAUDE_API_KEY` on the server; never written to any file |
| Tasks (registered before generation) | vector_add/fp16 `n=20971520` (single pass); softmax/fp16 `2048 x 10240` (row reduction); histogramming/int32 `N=67108864, num_bins=4096` (multi-kernel: scratch fill + private counting + reduce) — the representative case of each operator's real config |
| Trajectories | 3 tasks x {triton, cutile} x {gpt, claude} = 12, ten rounds each, ≤3 generations per round, repair only on a confirmed violation |
| Evaluation | isolated worker under bubblewrap (home hidden, no network, host read-only), device lock, three numerical checks against `impl_torch` (fresh / same-address refill / repeat), 1 warmup + 3 timed launches with CUDA-graph replay and 253 MB LLC eviction before each launch, three raw samples + mean |
| Timing of the runs | 2026-10-05 06:39 – 09:14 (dgx003 clock); the twelve trajectories ran as parallel processes, GPU evaluations serialized on the device lock |

## 2. Per-trajectory results

| task | model | status | rounds | valid | attempts | repairs | best ms (round) | samples | tokens (in/out/reasoning) | exact | reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cutile/histogramming/int32 | claude | complete | 10 | 10 | 10 | 0 | 0.1030 (10) | 0.1033, 0.1031, 0.1026 | 260928/286645/None | True | - |
| cutile/softmax/fp16 | claude | complete | 10 | 7 | 10 | 0 | 0.0167 (9) | 0.0168, 0.0167, 0.0166 | 252472/154841/None | False | - |
| cutile/vector_add/fp16 | claude | complete | 10 | 10 | 10 | 0 | 0.0218 (1) | 0.0218, 0.0218, 0.0217 | 231538/36783/None | True | - |
| cutile/histogramming/int32 | gpt | complete | 10 | 10 | 10 | 0 | 0.0918 (10) | 0.0923, 0.0920, 0.0911 | 158739/49967/42368 | True | - |
| cutile/softmax/fp16 | gpt | complete | 10 | 9 | 10 | 0 | 0.0171 (8) | 0.0175, 0.0170, 0.0169 | 156343/50325/44052 | True | r2:compliant, r3:compliant |
| cutile/vector_add/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0217 (8) | 0.0216, 0.0217, 0.0219 | 149208/29534/26185 | True | r5:compliant |
| triton/histogramming/int32 | claude | complete | 10 | 7 | 10 | 0 | 0.0751 (5) | 0.0762, 0.0746, 0.0744 | 261004/538209/None | True | r2:compliant, r3:compliant |
| triton/softmax/fp16 | claude | complete | 10 | 10 | 10 | 0 | 0.0162 (1) | 0.0167, 0.0159, 0.0161 | 274932/368662/None | True | r1:compliant, r9:compliant |
| triton/vector_add/fp16 | claude | complete | 10 | 9 | 10 | 0 | 0.0205 (7) | 0.0182, 0.0215, 0.0216 | 238080/176547/None | True | - |
| triton/histogramming/int32 | gpt | complete | 10 | 10 | 10 | 0 | 0.0699 (7) | 0.0711, 0.0694, 0.0691 | 162041/57960/48995 | True | - |
| triton/softmax/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0165 (10) | 0.0165, 0.0165, 0.0164 | 162267/44595/37016 | True | r1:compliant, r2:compliant |
| triton/vector_add/fp16 | gpt | complete | 10 | 10 | 10 | 0 | 0.0219 (10) | 0.0219, 0.0219, 0.0219 | 151350/26928/23251 | True | - |

Columns: rounds closed; valid rounds; attempts (generations); repairs
(same-round regenerations after a confirmed violation); best valid mean of
three timed launches and its round; the three raw samples of that round;
logical tokens (input / output / reasoning where the provider reports it);
whether every attempt's cost is known; recorded review decisions.

Every request's exact prompt, the provider's raw response (ids, terminal
status, raw usage), the candidate file, the compliance evidence, the worker
result and the archived sandbox evidence are under the trajectory directory
(see `docs/llm_v2/ARTIFACT_POLICY.md` for the layout).

## 3. Resume evidence (same trajectory, vector_add / triton / gpt, id `cd2b56f00ad1738f27a2`)

1. Round 1 ran with `--stop-after-rounds 1` (`logs/triton_gpt_vector_add_r1.log`): one request `resp_05a0e1d2…`, one evaluation, state saved, process exited with `paused`.
2. `--resume --stop-after-rounds 3` (`..._r3.log`): the runner reported `resuming cd2b56f0…: status in_progress, 1 rounds closed, 1 transport events`; rounds 2–3 requested; the ledger then held exactly three rows with three distinct response ids and round 1's row unchanged (`usage.jsonl`, `transport.jsonl`: `sending/succeeded` pairs for rounds 1, 2, 3 only).
3. `kill -9` test (`..._r4kill.log`): the process was killed immediately after `round 4 attempt 1: verdict clear`, i.e. after `response.json` was archived and the state saved but before the evaluation. The saved state showed round 4 pending with its response archived and no evaluation.
4. `--resume` (`..._final2.log`): the runner reused the archived round-4 response (`archived response reused (no request)`), ran the evaluation, and continued to round 10. `usage.jsonl` kept one row for round 4 (`resp_096bd7a1…`), `transport.jsonl` shows no second `sending` for round 4.

A second kind of resume was exercised on the cuTile softmax trajectories
(§5): `--resume --retry-incomplete` re-opened a round closed by an
evaluation-side infrastructure failure and re-evaluated the already
archived candidate without any new request.

## 4. Compliance reviews (human-visible decisions; all recorded in `reviews.jsonl` + `trajectory.json.notes`)

| Trajectory | Round | Evidence that blocked | Decision (reviewer) | Basis |
|---|---|---|---|---|
| softmax / cutile / gpt | 2 | missing `-inf` evidence | compliant (operator) | whole row held on chip (8192 + 2048 per row), no partial chunk; contract permits whole-row evaluation |
| softmax / cutile / gpt | 3 | missing `-inf` evidence | compliant (recheck under revised softmax rules) | rule entry removed; see §6 |
| vector_add / cutile / gpt | 5 | no kernel-level load/store evidence | compliant (operator) | loads/stores via cuTile raw-memory `load_offset`/`store_offset` inside `@ct.kernel` (documented API); evidence patterns extended |
| histogramming / triton / claude | 2 | `data_ptr` used as a key | compliant (operator) | `assert x.data_ptr() == input.data_ptr()` is a no-op contiguity guard, no cache |
| histogramming / triton / claude | 3 | `data_ptr` used as a key | compliant (recheck under the refined heuristic) | same guard line |
| softmax / triton / claude | 1 | missing `-inf` evidence | compliant (recheck after the checker fix) | `float("-inf")` is a string literal that the checker had blanked; strings now kept for evidence |
| softmax / triton / claude | 9 | missing `-inf` evidence | compliant (recheck) | the process had started before the softmax rule revision and still applied the removed entry |
| softmax / triton / gpt | 1 | missing `-inf` evidence | compliant (operator) | whole row held on chip (5 x 2048), exact tiling, fp32 statistics |
| softmax / triton / gpt | 2 | missing `-inf` evidence | compliant (recheck under revised softmax rules) | rule entry removed |

No candidate was found to violate a contract; no same-round repair was
triggered in the campaign (repairs = 0 everywhere). Reviews were decided by
the campaign operator (this Claude Code session), not by an LLM reviewer,
and are listed for the owner's veto (`DECISIONS_REQUIRED.md` E11).

## 5. Round outcomes and failures (round consumed, feedback given, no repair)

120 rounds in total: 112 `valid`, 5 `format_error`, 2 `runtime_error`, 1 `numerical_error`;
120 generations (one per round; no same-round repair was ever triggered);
121 transport attempts (120 succeeded, 1 failed); 2,458,902 logical input
tokens and 1,820,996 logical output tokens over the campaign.

- `format_error` (5, all Claude): the fenced block lacked `title="impl_<dsl>.py"` (histogramming/triton rounds 1, 6, 8; vector_add/triton round 1; softmax/cutile round 10). The strict parser consumed the round and the next prompt reported it.
- `runtime_error` by wall-clock (2): cuTile candidates that hold large whole-row tiles produced kernels whose `tileiras`/`ptxas` compile did not finish (one 20 MB PTX kept `ptxas` at 100 % CPU for 30 minutes while holding the device lock). Both first hit the old launcher's 1800 s limit and were recorded as `infrastructure_incomplete`; they were re-opened with `--resume --retry-incomplete` under the phase-aware launcher and classified as the candidate's `runtime_error` ("exceeded the evaluation wall-clock limit of 600 s during phase candidate_loaded"); both trajectories continued (softmax/cutile/claude round 4, softmax/cutile/gpt round 3; E12).
- `numerical_error` (1): softmax/cutile/claude round 8 (0.3 % of elements outside tolerance on the fresh-storage check).
- Transport: one Anthropic `overloaded_error` delivered as a stream error event (softmax/cutile/claude round 5, transport attempt 1); the SDK raised it as an `APIStatusError (200)` after the stream had opened, so it is recorded with `charged: unknown`; the retry succeeded. That attempt's cost is therefore unknown and the trajectory's cumulative token count is a lower bound (`cost_exact: False` in the table) — the accounting rule of R6 applied to a real event.

## 6. Rule and checker revisions made during the campaign (evaluator-only; recorded here and in git)

1. `required_evidence` patterns are now matched on comment-stripped text that keeps string literals (`contract_checks/2026-10-05.2`).
2. cuTile raw-memory `load_offset` / `store_offset` / `get_raw_memory` added to the kernel-level load/store evidence of 13 rule files.
3. softmax: the "`-inf` literal" required-evidence entry removed (the contract permits whole-row evaluation without partial chunks; masking is verified by execution).
4. static `data_ptr` heuristic: only key/stored uses are flagged.
5. launcher: phase-aware timeout classification; `--retry-incomplete`.

No contract text, Skill text or prompt template was changed in response to
any candidate's performance.

## 7. What this campaign does not establish

- No formal E(B) numbers: contracts are unapproved, assets are drafts, models are `candidate`.
- No cross-device claim: B200 only.
- No distillation: validation runs are refused as sources by code (`distill` on this campaign returns REFUSED; `outputs/llm_v2/evidence_2026-10-05/distill_refusal_*.stderr`).
