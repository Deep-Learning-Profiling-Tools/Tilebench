# FREEZE_DECISIONS — what is decided, what is fixed in code, what the owner still rules on

State after the freeze-preparation round (REVIEW_4a1f4591.md / NEXT_STEP_CLAUDE.md,
2026-10-05). `S_llm` is **not** frozen by this document; it lists the inputs a
freeze needs and separates them by who decides.

## A. Already decided by the study owner (provenance recorded; not re-opened)

| Decision | Where recorded | Evidence |
|---|---|---|
| Generators: `gpt-6.1-sol` (`reasoning.effort=xhigh`, `max_output_tokens=128000`, `OPENAI_API_KEY`) and `claude-opus-5-5` (`output_config.effort=xhigh`, NOT `max`; adaptive thinking; `max_tokens=128000`; `CLAUDE_API_KEY`) | `manifests/models.yaml` generator entries: `status: approved`, `approved_by` cites NEXT_STEP_CLAUDE.md 2026-10-05 §一 | `artifacts/.../evidence/probes/*_probe.json`; 120 archived responses echo these ids/settings |
| One DSL / one dtype / one fixed case per task; one deterministic configuration per generation, no hidden autotuner; stages may carry different fixed configurations; ten rounds; ≤3 generations per round, repair only on a confirmed violation | `manifests/study.yaml` (`trajectory`), `prompts/templates/system_interface.md`, `validation/*`, `evaluation/worker.py` | validation campaign: 120 rounds, 0 repairs, interface/config checks exercised |
| Runtime-only feedback; 1 warmup + 3 timed launches; complete operator timed (GPU time of every kernel launched by `run()`) | `study.yaml:timing`, `prompts/feedback.py`, `evaluation/timing.py` | 112 Proton profiles re-summed by the reviewer |
| Shared framework, loader, prompts and live wiring live on `exp/llm`; device branches only run them | `docs/llm_v2/RUNBOOK.md` (same `base`/`enhanced` commands on every device) | — |
| Publication of every round's shareable text artifact through the tracked `artifacts/llm_v2/` path (not only winners) | `docs/llm_v2/ARTIFACT_POLICY.md`, `.gitignore` (published Proton profiles re-included) | commits 4a1f4591, 7af863a7; clean-archive INDEX check in `tests/llm_v2/test_freeze_fixes.py` |
| The 2026-10-05 validation campaign stays unscored, is never a Base result and never a distillation source | `study.yaml:run_types`, `distillation/access.py` (validation refused by code) | `evidence/gates/distill_refusal_*` |

## B. Code-consistency items closed in this round (regression tests in `tests/llm_v2/test_freeze_fixes.py`)

| Item | Fix | Test |
|---|---|---|
| Publication integrity | 112 `.hatchet` profiles tracked (precise `.gitignore` exception); INDEX verified in a clean `git archive` | `test_published_index_verifies_in_a_clean_git_archive` |
| R6 resume binding | evaluator fingerprint (job tolerance/rules/timing/capture policy, checker + evaluation sources, worker timeout, isolation backend, environment) recorded at creation; formal resume refuses any difference, validation accepts only with `--allow-evaluator-change` and records it | `test_r6_resume_is_bound_to_the_evaluator_fingerprint` |
| Append-only evidence | first `compliance.json` carries candidate sha256 / checker / rules sha256 and is never rewritten; rechecks are `compliance_recheck_NNNN.json`; evaluations live in `eval_NNNN/` with META (reason, supersedes, executor, fingerprint); `re-evaluate` adds revisions to archived candidates | `test_evaluation_revisions_are_append_only_*`, `test_review_recheck_files_are_numbered_*` |
| R3/R4/R5 transport | attempt transport rebuilt from every durable event (monotonic ids; orphaned persisted once; lost responses keep their known charge; unknown charges propagate); explicit `--resume-transport --reason` with a durable `reopened` event; refusal reopen records settings hashes | `test_r3_*`, `test_r4_*`, `test_lost_response_*`, `test_r5_*` |
| R2 configuration snapshots | canonical JSON copies per read; three reads compared by value | `test_r2_config_snapshot_*` |
| R1 denominators | pre-declared eligibility only; incomplete/missing/unknown/SOL-unavailable stay in the denominator (`mean` None, `lower_bound_mean`, separately named `completed_only_mean`) | `test_r1_incomplete_task_stays_in_the_frozen_denominator` |
| R7 distillation sources | folds recomputed from the frozen manifest, state labels cross-checked, incomplete/validation refused, coverage against the pre-declared task set | `test_r7_*` |
| R8 synthesis | idempotent by input identity; truncated/interrupted responses kept as partial records with cost, never observations/skills; full request/raw response archives; per-attempt materials with diffs, SOL input | `test_r8_*` |
| R9/R10 Enhanced | optimization-skill provenance manifest validated (DSL, version, source device, mode, folds, source ids, content hash); component hash over the complete injected text incl. attachments | `test_r9_*`, `test_r10_*` |
| Isolation | bounded allowlist sandbox (system dirs, runtime prefix, repository with data masks, task reference only); sentinel probe; formal preflight requires the sandbox and a passing probe; AMD/Neuron device binding marked pending | `test_restricted_paths_are_unreadable_inside_the_sandbox`, `test_formal_preflight_requires_the_sandbox` |

## C. Fact checks and owner rulings still required before `S_llm`

| # | Question | Evidence to use | Recommendation (not applied) | Effect on the method |
|---|---|---|---|---|
| C1 | The eight `needs-review` contracts (batched_matmul, matmul_fp32_fp16_fp8, matmul_int8, streamk_matmul: prepacked B / F-Q boundary; destindex: output policy; gaussian_blur: tap precision; quantize_global: plain cast; radix_sort: pass count) and the 37 draft audits | `docs/llm_v2/CANONICAL_AUDIT.md`, `contracts/data/<op>/audit.json`, the manual kernels at `S_main` `ea04fb36` (read the fixed commit, not a summary) (2026-10-06: every contract is at **revision 2**, wording only: aligned≠mandatory, softmax degeneration, logical traversals, device-work timing boundary, fixed shape; `docs/llm_v2/CONTRACT_REVISION_2.md` lists the per-operator changes and 14 reviewer-flagged items that need a ruling) | A1: require the generated `run()` to accept B as given and pay any repack inside the timed call; mark the manual Triton references `human_reference_not_comparable` for those four operators. A4/A5: keep `S_main` semantics (plain cast; private-row histogram). A6: fresh outputs. A7: fp32 per-tap product | fixes what the 45 contracts demand; approval sets `status: approved` per audit.json |
| C2 | Scoring ceiling and F/Q. **Superseded in part on 2026-10-06**: the ceiling is now the empirical profile (`docs/llm_v2/CALIBRATION.md`; B200 measured and registered as `candidate`), the per-operator declaration is `arithmetic_modes.yaml` revision 2 (`proposed`), and the F/Q questions are the explicit decisions M1–M4 of §D. The datasheet table is a reference only. | `arithmetic_modes.yaml` (decisions block), `artifacts/llm_v2/scoring/B200/*.json` (110 tasks, all `definition_pending` with provisional T_emp), `artifacts/llm_v2/calibration/B200/B200-20261006T063415Z-39b55bd3/` | freeze the B200 profile (`calibration-register --status frozen --by`), approve M1 and M2, decide M3/M4; keep the frozen F/Q otherwise | every task's T_emp; nothing is scored until M1 is approved |
| C3 | Fold assignment (`folds.yaml` `proposed`, families whole, 15/15/15) | `manifests/folds.yaml` | freeze as proposed unless a family must be split | distillation scope; cannot change after any Base result is seen |
| C4 | Distiller role: model id, settings, output cap; whether the same two generators or one of them | `models.yaml` `distiller: unset` | `gpt-6.1-sol` xhigh / 128000 for both map and reduce (single model keeps the skill text attributable); record as a decision, not a default | `distill` refuses until set |
| C5 | Sharing grants for TileLang (internal, unlicensed local guide) and NKI (private) bodies; repository LICENSE | `skills/manifest.json` grants, `DECISIONS_REQUIRED.md D1–D3, D6` | no grant until the source licence is confirmed; TileLang/NKI runs blocked by the loader until then | TileLang and Trn2 campaigns cannot start |
| C6 | Capture-failure policy for formal runs (`time_eagerly_and_flag` today) | `study.yaml:timing.capture_failure_policy`, E8 | formal: record a failed capture as `timing_error` (round consumed) so every valid sample of a device shares one execution mode | changes the valid-round set on NVIDIA devices |
| C7 | Evaluation wall-clock limit per candidate | E12; the two cuTile compiles that hit 1800 s | fix the limit in `study.yaml` (e.g. 900 s) before the formal campaign; same value for every candidate of a campaign (no per-candidate shortening) | `runtime_error` outcomes; fingerprint element |
| C8 | Review authority for `review_required` verdicts in formal runs (operator decision with recorded evidence vs. a second reviewer / LLM reviewer) | E11; `reviews.jsonl` of the validation campaign | keep operator decisions with recorded evidence; add a second reviewer only if the study plan requires it; no LLM reviewer by default | affects which trajectories can be unblocked and by whom |

Nothing in C was changed by code in this round. `approved_by` fields, fold
status, grants and `S_llm` are set only by the owner.

## D. Round of 2026-10-06 (empirical calibration; `NEXT_STEP_EMPIRICAL_CALIBRATION.md`)

| # | Item | State | Owner action |
|---|---|---|---|
| D1 | Empirical calibration entry point, profile schema, protocol v2, SOL loader (`metrics/empirical.py`), campaign binding and gates | implemented; `tests/llm_v2/test_calibration.py` | — |
| D2 | B200 profile `B200-20261006T063415Z-39b55bd3` (10 modes calibrated; power-capped sustained rates; raw samples tracked) | registered `candidate` in `manifests/calibration.yaml` | freeze: `calibration-register --device B200 --profile <...>/profile.json --status frozen --by "<owner>"` |
| M1 | `arithmetic_modes.yaml` revision 2: per-operator modes, f_kind/q_kind (no dtype-wide defaults) | `proposed` | approve (`status: approved`, `approved_by`) or amend entries |
| M2 | `memory_only` target for element/comparison/conversion/integer tasks (33 of 110 B200 tasks) | `proposed` | approve, or ask for integer/compare/convert probes whose operation definition matches each F |
| M3 | flash_attention F: dense count kept vs causal override `2*B*H*D*S*(S+1)` | `proposed`, not applied | approve the override or keep dense (then the audit must say "dense-equivalent") |
| M4 | radix_sort Q: 16-pass algorithm-specific model requires the contract to fix the digit width (C1 item) | `proposed` | decide with C1 radix_sort |
| C1' | gaussian_blur fp16 tap precision (declared fp32_fma_vector pending) | open | decide (A7) |
| D3 | Contract revision 2 reviewer-flagged items (14, `CONTRACT_REVISION_2.md` §Items) | open | rule per item; `contract_revision` stays 2 until approval |
| D4 | GH200 / MI300X / Trn2 profiles | none | run `calibrate` locally with this commit (Trn2: native adapter from the NKI window) |

Formal B200 Base cannot start until D2 (frozen), M1 (+M2 for memory_only tasks, M3/M4/C1' for their tasks), C1 (approved contracts), D4-independent skill approvals (DECISIONS_REQUIRED D4) are done; `preflight --devices B200 --live --run-type formal` names each missing item.
