# READINESS — TileBench++ LLM protocol v2, shared-framework phase (2026-10-05)

Branch `exp/llm` (worktree `../llm_wt`), cut from `S_main = ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`
(main after PR #318). This phase delivers the v2 plan's stages B–F as
testable code and reviewed assets. It does **not** run a Base campaign,
distill a real Optimization Skill, create device branches, or declare
`S_llm` frozen.

## 1. What is implemented and tested

| Stage | Deliverable | State |
|---|---|---|
| B. Task contracts | 45 × `contract.md` + `evaluator_rules.json` + `audit.json`; 90 independent extractions; `CANONICAL_AUDIT.md` | 37 `draft`, 8 `needs-review`, 0 `approved` (owner act) |
| C. Framework | `tilebench.llm.v2` (manifests, tasks, skills, prompts, providers, validation, metrics, orchestration, evaluation, distillation, contracts, CLI) | implemented; 93 CPU/mock tests |
| D. Reference + Device | Triton 3.6.0 / cuTile 1.5.0 / TileLang 0.1.11 references (section-revised, introspection-verified, REVISION_MAPs); 4 device snapshots with sources | all `draft`; NKI reference private/metadata only |
| E. Experiment manifests | `study.yaml`, `models.yaml`, `folds.yaml` (proposed), `arithmetic_modes.yaml` (proposed), `skills/manifest.json`, task table (880 tasks) | validated; live-blocking items listed in DECISIONS_REQUIRED |
| F. Pre-run integration | mock ten-round trajectory with persistence/resume; dry-run of 770 prompts on 8 device/DSL pairs; prompt snapshots from the real renderer; B200 worker smoke (positive + cheat) | done on this host only |

### Test evidence (this host, dgx003, `tilebench_env`)

```
python -m pytest tests/llm_v2 -q       -> 93 passed
python -m pytest tests -q              -> 570 passed, 2 skipped (pre-existing skips), 0 failed
python -m compileall -q tilebench scripts tests -> ok
python -m tilebench.llm.v2 validate-manifests -> errors: [], contracts draft 37 / needs-review 8
python -m tilebench.llm.v2 dry-run --device <D> --dsl <L> --allow-draft (8 pairs)
   B200 triton/cutile/tilelang 110/110/110 rendered; GH200 110/110/110; MI300X 109 + 1 unsupported;
   Trn2 110 needs_review (no prompt rendered); max prompt 57,529 chars (limit 260,000)
```

Coverage of the ten required test areas (`tests/llm_v2/`):
1. single-DSL single-task requests + strict parser; no cross-task context (`test_state_machine_and_runner.py::test_no_cross_task_context_in_prompts`, `test_validation.py`)
2. missing/hash/version/permission failures; Base/Enhanced differ only by the Optimization component (`test_skills_loader.py`)
3. no scoring/manual leakage in config, feedback, diagnostics, contracts (`test_prompts.py`, `test_contracts_and_campaign.py`, `test_manifests_and_tasks.py`)
4. case selection from expanded cases; family folds; target/held-out access boundaries (`test_manifests_and_tasks.py`, `test_distillation.py`)
5. ten-round state machine, ≤3 generations, ordinary failures without same-round repair, review_required, infrastructure incomplete (`test_state_machine_and_runner.py`)
6. cached/reasoning normalization, failed-attempt charging, unknown usage, resume without re-request (`test_providers_and_usage.py`, runner tests)
7. 1 warmup / 3 timed call counts, three raw samples + mean, graph vs eager, actual capture status (`test_evaluation.py`)
8. fresh-input / same-address anti-cache, mutation restore, worker isolation (`test_evaluation.py`; GPU evidence in `docs/llm_v2/evidence/b200_smoke_2026-10-05/`)
9. E(B) zero, >1 flagged, best never erased, non-uniform token steps, operator-balanced aggregation, incomplete vs unsupported (`test_metrics.py`)
10. same-fold skill hash identical across devices, NKI source = Trn2 only, release scope ≠ evaluation scope (`test_skills_loader.py`, `test_distillation.py`)

Not covered by tests: delegation/hidden-autotuning detection is covered
statically and by the GPU cheat smoke, not by a GPU test; a worker crash
is simulated through the `infrastructure_incomplete` path (the launcher's
subprocess timeout/crash branch is exercised only by the smoke).

## 2. Readiness per device / DSL

| Device / DSL | Reference | Device ctx | Contracts | Eligibility | Timing adapter | Model cfg | Live state |
|---|---|---|---|---|---|---|---|
| B200 triton/cutile/tilelang | draft (introspection-verified here) | draft, verified on device | 37 draft / 8 needs-review | 330 eligible | `proton_cuda_graph`, smoke-tested | unset | **blocked** (approvals, model ids) |
| GH200 triton/cutile/tilelang | same text (same hash) | draft, from archived captures, not verified on device | same | 330 eligible | same adapter, not run there | unset | **blocked** + remote preflight pending |
| MI300X triton | same | draft, from archived captures | same | 109 eligible, 1 unsupported (fp8) | `proton_rocm_eager`, not run there | unset | **blocked** + remote preflight pending |
| Trn2 nki | private, metadata only | draft, vendor docs only, no capture | n/a until NKI reference exists | 110 needs_review | `neuron_runtime_trace` **not ready** | unset | **blocked** (NKI_HANDOFF) |

`python -m tilebench.llm.v2 preflight --live` reports these blockers per
pair; `base`/`enhanced`/`distill` refuse to start on them.

## 3. Blockers (exact)

1. Contract approval: 0/45 approved; 8 need a decision (see `CANONICAL_AUDIT.md` and `DECISIONS_REQUIRED.md §A`).
2. `models.yaml`: generator/distiller model ids, reasoning settings, max output tokens unset.
3. `folds.yaml` status `proposed`; `arithmetic_modes.yaml` status `proposed`; peak table gaps (`DECISIONS_REQUIRED.md §B`).
4. Skill approvals: references and device snapshots are `draft`; TileLang reference permission `internal` pending licence confirmation; repository has no LICENSE.
5. Remote evidence: GH200 / MI300X / Trn2 have run nothing of v2 (`RUNBOOK.md` "Remote devices").
6. NKI: adapter not validated, reference not derivable until permissions are confirmed (`NKI_HANDOFF.md`).
7. Live runner wiring (provider + `SubprocessEvaluator` per device) is intentionally not attached to the `base`/`enhanced` commands on the shared branch.

## 4. Protocol deviations introduced by this phase

None intended. Items where the implementation had to pick a reading of the v2
text are listed in `DECISIONS_REQUIRED.md §E` (positional `run()` call,
graph-capture failure handling, undetermined-cost rule, review gating).
The MI300X `fp8_e4m3fn` task is marked `unsupported` from the manual
campaign's evidence rather than from a generation attempt.

## 5. S_llm

Not proposed. Conditions before an `S_llm` candidate can be named:
contracts approved (or needs-review resolved), folds and arithmetic modes
frozen, skills approved, model ids set, remote preflights returned with
evidence.
