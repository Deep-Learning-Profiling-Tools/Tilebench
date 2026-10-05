# READINESS — TileBench++ LLM protocol v2 (state after the review-fix round, 2026-10-05)

Branch `exp/llm` (worktree `../llm_wt`), cut from `S_main = ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`.
The shared framework (89140bb3) was reviewed (REVIEW_89140bb3.md, findings
R1–R12); this round fixed every finding, wired the live execution chain into
the shared CLI, and ran a real ten-round validation campaign on B200 with
GPT and Claude. It does **not** run a formal Base campaign, distill a real
Optimization Skill, approve contracts or skills, create device branches, or
declare `S_llm` frozen.

## 1. Three states

### Implemented (code, tests, docs)

| Finding | Fix | Where | Regression |
|---|---|---|---|
| R1 gate-only CLI | `base`/`enhanced` run manifests → task context → provider factory → `TrajectoryRunner` → `SubprocessEvaluator` → state/ledger/artifacts; `distill` runs the real map/reduce with trusted index and persistence | `orchestration/campaign.py::run_campaign`, `cli.py` | real campaign (§2) |
| R2 missing atol/rtol | evaluator-only `EvaluationJob` (effective tolerance with arch override, rules, normalized timing, identity); the campaign asserts prompt tolerance == evaluator tolerance | `evaluation/job.py` | `test_review_fixes.py::test_evaluation_job_*`, `test_prompt_and_evaluator_tolerance_agree` |
| R3 regex false positives / alias miss | import-alias resolution, kernel-vs-host scope, regex on comment/string-stripped code, confirmed only with AST corroboration, rule `scope` on all 250 rules | `validation/static_checks.py`, `contract_checks.py`, `contracts/data/*/evaluator_rules.json` | `test_kernel_arithmetic_is_not_host_delegation`, `test_import_alias_autotune_is_confirmed`, `test_regex_rules_ignore_comments_strings_and_kernel_scope` |
| R4 interface not enforced | worker requires callable `run`/`get_last_config` (dict, JSON, identical across three reads), scans for autotuner objects, `interface_error` status; prompt text fixed (literals vs derived quantities) | `evaluation/worker.py`, `prompts/templates/system_interface.md` | `test_worker_requires_exports_and_fixed_config`, `test_worker_detects_autotuner_object_and_output_aliasing` |
| R5 providers | streaming, `api_key_env` (`CLAUDE_API_KEY`), `ProviderConfigError` never retried, transport failures classified `charged: no/unknown`, terminal status/truncation, factory from models.yaml | `providers/*.py` | `test_provider_kwargs_carry_the_configured_settings`, `test_sdk_exceptions_are_classified`, `test_factory_refuses_without_key_env`, live probes (§2) |
| R6 ledger gaps | per-transport-attempt charge accounting, attempt cost unknown when any charge is unknown, prompt_too_long = 0/not_sent, reconciliation of the response→ledger crash window, orphaned `sending` events | `orchestration/runner.py`, `state.py` | `test_unknown_transport_charge_propagates_to_the_attempt`, `test_prompt_too_long_*`, `test_crash_between_response_and_ledger_is_reconciled_on_resume`, `test_orphaned_sending_event_*` |
| R7 unknown → 0 | `efficiency_at` None for unknown cost / no SOL / incomplete; aggregate `mean` None with a labelled `lower_bound_mean`; finite-sample checks; coverage-checked pairing | `metrics/efficiency.py`, `state_machine.py` | `test_unknown_cost_and_missing_sol_are_not_zero`, `test_non_finite_latency_*`, `test_valid_round_needs_consistent_finite_samples` |
| R8 distillation binding | trusted index with file hashes, root containment, symlink refusal, state identity re-verification, validation runs refused, persisted/resumable observations, template-content hash | `distillation/access.py`, `orchestrator.py` | `test_distillation_refuses_identity_mismatch_escape_and_validation_runs`, `test_distillation_persists_and_reuses_observations` |
| R9 internal ≠ sendable | `sendable_to` per provider and `publishable` per asset, independent of status; TileLang/NKI carry no grant | `skills/loader.py`, `skills/manifest.json` | `test_provider_grants_are_separate_from_status`, `test_real_manifest_grants_match_permissions` |
| R10 timing names / text | `timing_settings()` normalization, versioned `capture_failure_policy`, `timing_mode_differs` flag, restore hook before capture-prep launches, prompt describes the real check/timing protocol and GPU-time measurement | `evaluation/job.py`, `timing.py`, `study.yaml` | `test_timing_settings_normalize_study_keys`, `test_evaluation.py` |
| R11 oracle alias | reference and candidate outputs frozen before restore; mutation judged per declared index; output aliasing detected | `evaluation/anticache.py` | `test_reference_returning_mutated_input_is_frozen_before_restore` |
| R12 isolation / publication | bubblewrap sandbox (home hidden, no network, read-only host, private tmp) with an honest fallback report; sandbox evidence archived before deletion; `export-publication` + ARTIFACT_POLICY | `evaluation/launcher.py`, `orchestration/publication.py`, `docs/llm_v2/ARTIFACT_POLICY.md` | `test_isolation_report_and_bwrap_argv`, `test_publication_withholds_non_publishable_context`, live sandbox probe |
| item 11 (violation fallback) | a round closed by three violations is shown by outcome only; the last compliant implementation is the base | `runner._round_view`, `renderer._prev_block` | `test_execution_confirmed_violation_triggers_repair_and_violation_round_hides_code` |

Test evidence (dgx003, `tilebench_env`, final code state): `python -m pytest tests/llm_v2 -q` → 128 passed;
`python -m pytest tests -q` → 605 passed, 2 skipped (pre-existing skips); `python -m compileall -q tilebench scripts tests` → ok
(logs under `outputs/llm_v2/test_logs/`, copied into the handoff).

### Executed (real models, real device)

See `docs/llm_v2/VALIDATION_2026-10-05.md` for the per-trajectory table
(request ids, usage, three timing samples, reviews, artifact paths).
Campaign `validation_b200_2026-10-05`, run type `validation` (unscored),
condition Base, B200, DSLs Triton and cuTile, models `gpt-6.1-sol`
(reasoning.effort xhigh, max_output_tokens 128000) and `claude-opus-5-5`
(output_config.effort xhigh, adaptive thinking, max_tokens 128000), tasks
vector_add/fp16 (single pass), softmax/fp16 (row reduction),
histogramming/int32 (multi-kernel: scratch fill + private counting +
reduce), representative cases from the operators' real configs. Twelve
trajectories, ten rounds each, isolated evaluation (bwrap), 1 warmup + 3
timed launches with graph replay. Resume was exercised twice on the same
trajectory: a pause/resume at round boundaries and a `kill -9` after a
response was archived but before its evaluation (the archived response was
reused without a new request; the usage ledger kept one row per response).

### Still blocked (owner decisions; nothing was decided by the framework)

1. Contracts: 0/45 approved (37 draft, 8 needs-review); `DECISIONS_REQUIRED.md §A`.
2. Models: `candidate` status only; formal runs need `approved` (§C1); distiller unset.
3. Folds and arithmetic modes `proposed`; peak table gaps (§B).
4. Skills `draft`; grants set for public assets, none for TileLang/NKI (§D).
5. Remote devices: GH200 / MI300X / Trn2 have run nothing of v2; the same
   `base` command runs there once the branch is checked out (RUNBOOK).
6. NKI adapter not validated (NKI_HANDOFF).
7. Protocol readings E6–E11 (interface_error, execution-confirmed
   violations, capture-failure policy, run types, regex corroboration,
   review authority).

## 2. Readiness per device / DSL

| Device / DSL | Reference | Device ctx | Contracts | Timing adapter | Provider grant | Live chain | Formal gate |
|---|---|---|---|---|---|---|---|
| B200 triton / cutile | draft, public, granted | draft, verified on device | draft/needs-review | `proton_cuda_graph`, exercised in the validation campaign | openai + anthropic | **executed** (validation) | blocked (approvals) |
| B200 tilelang | draft, internal, **no grant** | same | same | same | none | blocked by grant | blocked |
| GH200 triton / cutile / tilelang | same texts | draft, not verified | same | same adapter, not run there | as above | not run | blocked |
| MI300X triton | same | draft, not verified | same | `proton_rocm_eager`, not run there | as above | not run | blocked |
| Trn2 nki | private, metadata only | draft, vendor docs | n/a | `neuron_runtime_trace` **not ready** | none | blocked | blocked |

## 3. Protocol deviations introduced by this round

None intended. Readings the owner must confirm are listed in
`DECISIONS_REQUIRED.md §E` (E6–E11). Two evaluator-rule revisions were made
during the validation campaign and are recorded there and in the campaign
report: cuTile raw-memory `load_offset`/`store_offset` added to the
kernel-level load/store evidence (13 rule files), and the softmax
"-inf literal" required-evidence entry removed (the contract permits
whole-row-on-chip evaluation without partial chunks). Both are
evaluator-only draft rules; no contract text, no Skill text and no prompt
template was changed in response to a candidate's performance.

## 4. S_llm

Not proposed. Conditions unchanged: contracts approved (or needs-review
resolved), folds and arithmetic modes frozen, skills approved, model ids
approved, remote preflights returned with evidence.
