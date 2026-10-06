# READINESS — TileBench++ LLM protocol v2 (state after the freeze-preparation round, 2026-10-05)

Branch `exp/llm` (worktree `../llm_wt`), cut from `S_main = ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`.
Rounds so far: shared framework (89140bb3) → review fixes R1–R12 + live
chain + real ten-round validation campaign (4a1f4591) → freeze preparation
(this round: REVIEW_4a1f4591.md counterexamples R1–R10, publication
integrity, evaluator binding, append-only evidence, transport recovery,
denominators, distillation/Enhanced boundaries, bounded isolation).
`S_llm` is **not** frozen; `docs/llm_v2/FREEZE_DECISIONS.md` lists what a
freeze still needs and who decides.

## 1. Three states

### Implemented and evidenced (code + regression tests)

| Area | What holds now | Where | Test |
|---|---|---|---|
| Publication integrity | the 112 archived Proton profiles are tracked (precise `.gitignore` exception, commit 7af863a7); INDEX.json of the published campaign verifies in a clean `git archive HEAD` (1,614 → 1,634 entries after the appended re-evaluations) | `.gitignore`, `artifacts/llm_v2/validation_b200_2026-10-05` | `test_published_index_verifies_in_a_clean_git_archive` |
| Evaluator binding on resume (R6) | fingerprint (job tolerance/rules/timing/capture policy/adapter, checker + evaluation + core timer/verifier sources, worker timeout, isolation backend, environment) recorded at creation; formal resume refuses any difference; validation accepts only `--allow-evaluator-change`, recorded in state and `evaluator_changes.jsonl` | `evaluation/fingerprint.py`, `orchestration/campaign.py::_check_resume` | `test_r6_resume_is_bound_to_the_evaluator_fingerprint` |
| Append-only evidence | first `compliance.json` with candidate sha256 / checker fingerprint / rules sha256, reused on replay, never rewritten; rechecks `compliance_recheck_NNNN.json`; evaluations `eval_NNNN/` with META (reason, supersedes, executor, fingerprint); `re-evaluate` appends to archived candidates; `retry_incomplete` produces a new revision | `orchestration/runner.py`, `cli.py::cmd_reevaluate` | `test_evaluation_revisions_are_append_only_*`, `test_review_recheck_files_are_numbered_*` |
| Transport recovery (R3/R4/R5) | attempt transport rebuilt from every durable event with monotonic ids; orphaned persisted once and reused; lost responses keep their known charge and are re-requested; unknown charges propagate; explicit `--resume-transport --reason` with a durable `reopened` event; refusal reopen records settings hashes; no round/candidate added | `orchestration/runner.py::durable_transport`, `campaign.py::reopen_transport_attempt` | `test_r3_*`, `test_r4_*`, `test_lost_response_*`, `test_r5_*` |
| Config snapshots (R2) | canonical JSON copies per read, compared by value, three reads recorded | `evaluation/worker.py::read_config` | `test_r2_config_snapshot_*` |
| Denominators (R1) | pre-declared eligibility separated from execution state; `mean` None when any eligible task is incomplete/missing/unknown; `lower_bound_mean` over the full denominator; `completed_only_mean` named separately | `metrics/efficiency.py::aggregate` | `test_r1_*` |
| Distillation sources (R7) | folds recomputed from the frozen manifest, state labels cross-checked, validation/incomplete/other-schema refused, coverage against the pre-declared task set (`--allow-partial-coverage` recorded) | `distillation/access.py`, `orchestrator.py::verify_state_identity`, `cli.py::cmd_distill` | `test_r7_*` |
| Synthesis (R8) | idempotent by input identity; truncated/interrupted responses are partial records with cost, never observations/skills; full request/raw response archives; per-attempt sources, diagnostics, diffs, raw samples and offline SOL input (versioned material rules) | `distillation/orchestrator.py` | `test_r8_*` |
| Enhanced boundary (R9/R10) | optimization-skill provenance manifest validated before injection; component hash over the complete injected text incl. attachments | `skills/loader.py` | `test_r9_*`, `test_r10_*` |
| Isolation | bounded allowlist bubblewrap sandbox (system dirs, runtime prefix, repository with masks over benchmarks/results/artifacts/outputs/skills/docs/tests/contracts/.git; only the task's `impl_torch.py` re-bound); sentinel probe; formal preflight requires bwrap + passing probe; NVIDIA nodes verified, AMD/Neuron described and pending | `evaluation/launcher.py` | `test_restricted_paths_are_unreadable_inside_the_sandbox`, `test_formal_preflight_requires_the_sandbox` |

Tests (dgx003, `tilebench_env`, this code state): `python -m pytest tests/llm_v2 -q` → 146 passed;
`python -m pytest tests -q` → 623 passed, 2 skipped; `python -m compileall -q tilebench scripts tests` → ok.
Logs: `artifacts/llm_v2/validation_b200_2026-10-05/evidence/test_logs_freeze/`; isolation probe and
formal preflight reports: `.../evidence/gates_freeze/`.

### Executed on hardware in this round (B200; no new model requests)

- Isolation probe inside the allowlist sandbox: manual DSL implementations,
  other operators' references, results, artifacts, skills, contracts, the
  sentinel under `outputs/`, the real home, `.git`, a sibling checkout and
  the network are unreachable; the task reference and the runtime are
  readable; no secret-named environment variable is present.
- Two archived validation candidates re-evaluated as append-only revisions
  (`eval_0002`, not adopted): vector_add/triton/gpt round 1 → valid, graph,
  0.023008 / 0.021823 / 0.022016 ms (original 0.022048 / 0.021888 / 0.022527);
  vector_add/cutile/gpt round 1 → valid, graph, 0.021984 / 0.021856 / 0.021984
  ms (original 0.021728 / 0.021952 / 0.021729). Both compiled and ran inside
  the allowlist sandbox; original records untouched.

The twelve real ten-round trajectories of 2026-10-05 were not re-run
(their archives were confirmed consistent by the reviewer); they remain
validation data, never Base or distillation input.

### Conditional capabilities pending hardware verification

- GH200 / MI300X / Trn2: nothing of v2 has run there. The same commands
  apply; the sandbox device binding for AMD (`/dev/kfd`, `/dev/dri`) and
  Neuron (`/dev/neuron*`) is described in `evaluation/adapters.py` and
  marked `pending` in every isolation report until exercised.
- The NKI timing adapter is not ready (NKI_HANDOFF).
- ROCm inside bubblewrap (roctracer/Proton) has not been exercised.

### Owner rulings still required (FREEZE_DECISIONS §C)

C1 eight needs-review contracts + 37 draft audits; C2 F/Q formulas and
per-device peaks/modes; C3 fold freeze; C4 distiller role; C5 TileLang/NKI
grants and repository licence; C6 formal capture-failure policy; C7 fixed
evaluation wall-clock limit; C8 review authority. Nothing in C was changed
by code; `approved_by`, fold status, grants and `S_llm` are the owner's.

## 2. Readiness per device / DSL

| Device / DSL | Reference | Device ctx | Contracts | Timing adapter | Sandbox | Provider grant | Live chain | Formal gate |
|---|---|---|---|---|---|---|---|---|
| B200 triton / cutile | draft, public, granted | draft, verified on device | draft/needs-review | `proton_cuda_graph`, exercised | allowlist bwrap, probe passing | openai + anthropic | **executed** (validation + appended re-evaluations) | blocked (C1–C3, C6–C8) |
| B200 tilelang | draft, internal, **no grant** | same | same | same | same | none | blocked by grant | blocked |
| GH200 triton / cutile / tilelang | same texts | draft, not verified | same | same adapter, not run there | not exercised there | as above | not run | blocked |
| MI300X triton | same | draft, not verified | same | `proton_rocm_eager`, not run there | AMD binding pending | as above | not run | blocked |
| Trn2 nki | private, metadata only | draft, vendor docs | n/a | `neuron_runtime_trace` **not ready** | Neuron binding pending | none | blocked | blocked |

## 3. Protocol deviations introduced by this round

None intended. Readings added for confirmation: `DECISIONS_REQUIRED.md`
E13 (one evaluator fingerprint per formal campaign; append-only
revisions) and E14 (transport reopen semantics). The generator decision is
recorded as the owner's (models.yaml `approved_by`), not as a framework
choice.

## 4. S_llm

Not proposed. Required before a candidate can be named: C1–C8 above
resolved, assets approved with grants, remote preflights returned with
evidence.
