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
| C1 | The eight `needs-review` contracts (batched_matmul, matmul_fp32_fp16_fp8, matmul_int8, streamk_matmul: prepacked B / F-Q boundary; destindex: output policy; gaussian_blur: tap precision; quantize_global: plain cast; radix_sort: pass count) and the 37 draft audits | `docs/llm_v2/CANONICAL_AUDIT.md`, `contracts/data/<op>/audit.json`, the manual kernels at `S_main` `ea04fb36` (read the fixed commit, not a summary) | A1: require the generated `run()` to accept B as given and pay any repack inside the timed call; mark the manual Triton references `human_reference_not_comparable` for those four operators. A4/A5: keep `S_main` semantics (plain cast; private-row histogram). A6: fresh outputs. A7: fp32 per-tap product | fixes what the 45 contracts demand; approval sets `status: approved` per audit.json |
| C2 | F/Q formulas versus the v2 "frozen compulsory traffic" rule (rope clone, streamk scratch, histogramming scratch, bitonic passes, flash_attention dense-vs-causal flops) and the peak table (B200.json tensor-core only; HGX page 2x; no fp32_vector/int_vector; no GH200/MI300X/Trn2 files) | `DECISIONS_REQUIRED.md §A2–A3, §B`, `tilebench/data/peak_performance/`, vendor pages | keep the frozen formulas for T_SOL (document the known discrepancies in the audit); add measured bandwidth + declared-mode peaks per device before any cross-device E(B); flash_attention: record both dense and causal F, score with the frozen one | T_SOL per task; tasks whose mode has no peak stay `peak_missing` (no E(B)) |
| C3 | Fold assignment (`folds.yaml` `proposed`, families whole, 15/15/15) | `manifests/folds.yaml` | freeze as proposed unless a family must be split | distillation scope; cannot change after any Base result is seen |
| C4 | Distiller role: model id, settings, output cap; whether the same two generators or one of them | `models.yaml` `distiller: unset` | `gpt-6.1-sol` xhigh / 128000 for both map and reduce (single model keeps the skill text attributable); record as a decision, not a default | `distill` refuses until set |
| C5 | Sharing grants for TileLang (internal, unlicensed local guide) and NKI (private) bodies; repository LICENSE | `skills/manifest.json` grants, `DECISIONS_REQUIRED.md D1–D3, D6` | no grant until the source licence is confirmed; TileLang/NKI runs blocked by the loader until then | TileLang and Trn2 campaigns cannot start |
| C6 | Capture-failure policy for formal runs (`time_eagerly_and_flag` today) | `study.yaml:timing.capture_failure_policy`, E8 | formal: record a failed capture as `timing_error` (round consumed) so every valid sample of a device shares one execution mode | changes the valid-round set on NVIDIA devices |
| C7 | Evaluation wall-clock limit per candidate | E12; the two cuTile compiles that hit 1800 s | fix the limit in `study.yaml` (e.g. 900 s) before the formal campaign; same value for every candidate of a campaign (no per-candidate shortening) | `runtime_error` outcomes; fingerprint element |
| C8 | Review authority for `review_required` verdicts in formal runs (operator decision with recorded evidence vs. a second reviewer / LLM reviewer) | E11; `reviews.jsonl` of the validation campaign | keep operator decisions with recorded evidence; add a second reviewer only if the study plan requires it; no LLM reviewer by default | affects which trajectories can be unblocked and by whom |

Nothing in C was changed by code in this round. `approved_by` fields, fold
status, grants and `S_llm` are set only by the owner.
