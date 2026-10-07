# S_llm — frozen shared framework of the formal LLM campaigns

## Current: protocol revision 4 (2026-10-07)

| item | value |
|---|---|
| S_llm commit | the merge commit of the shared-framework PR on `main`, tag `S_llm-rev4-20case-2026-10-07` (the tag records the exact SHA) |
| frozen hashes | `tilebench/llm/v2/manifests/frozen/B200_base_rev4_2026-10-07.json` (protocol identity revision 4: 20 cases per task, adjudicator `claude-code`, checker `contract_checks/2026-10-07.v4` with frozen `run()` input kinds; study config hash, case sets 45 x 20, 45 contracts (contract revision 3 where migrated), evaluator and core sources, Reference Skills triton/cutile/tilelang, B200 Device Context, empirical profile, 900-row scoring table, 900-case torch baseline, folds, model settings, transport, pricing snapshot `8eeac6c2...`, excluded campaigns); `tests/llm_v2/test_frozen_rev4.py` checks it |
| campaign branches | `exp/llm-b200`, `exp/llm-gh200`, `exp/llm-mi300x`, `exp/llm-trn2`, each brought to S_llm; no campaign runs from `exp/llm` or `main` |
| B200 campaign identity | `formal_b200_base_rev4_checkerv4_2026-10-07` (Triton, cuTile, TileLang x gpt, claude; Base; formal; 45 operators x 1 dtype x 20 cases); the first launch `formal_b200_base_rev4_20case_2026-10-07` (checker v3) is an excluded aborted campaign |
| scheduling | one `schedule` process for both models and all DSL tracks, adaptive concurrency from 3, global hard evaluation backlog 16, STOP-file pause |
| environment | dgx003, `LD_LIBRARY_PATH` unset, bwrap sandbox, worker wall-clock 3600 s per 20-case suite |
| validation smoke | `validation_b200_rev4_vector_add_1r_2026-10-07c` (vector_add, 1 round, 6 tracks, checker v4; not formal data) |
| protocol | docs/llm_v2/PROTOCOL_REVISION_4.md |

## Superseded: revision 3 (2026-10-06)

| item | value |
|---|---|
| S_llm commit | `38a925d87e97a5f5951c474d9d10a78bda0e40a9` (branch `exp/llm`, tag `S_llm-1dtype-5r-checkerv2-2026-10-06`) |
| frozen hashes | `tilebench/llm/v2/manifests/frozen/B200_base_rev3_2026-10-06.json` (study config hash `e2959a1b...`, protocol identity, representative-dtype manifest, checker `contract_checks/2026-10-06.v2`, evaluator sources, 45 contracts, Reference Skills triton/cutile/tilelang, B200 Device Context, empirical profile + 45-task scoring table, folds, model settings, transport, pricing snapshot `8eeac6c2...`, excluded campaigns); `tests/llm_v2/test_frozen_rev3.py` checks it |
| campaign branch | `exp/llm-b200-5r` cut from S_llm, worktree `../llm_b200_5r_wt` |
| campaign identity | `formal_b200_base_1dtype5r_2026-10-06` (Triton, cuTile, TileLang x gpt, claude; Base; formal; 45 operators x 1 dtype) — excluded pilot (manifests/excluded_campaigns.yaml) |
| scheduling | `schedule` (one process per model), adaptive concurrency from 3, STOP-file pause |
| environment | dgx003, `LD_LIBRARY_PATH` unset, bwrap sandbox, worker wall-clock 1800 s |
| protocol | docs/llm_v2/PROTOCOL_REVISION_3.md |

## Superseded: revision 2

| item | value |
|---|---|
| S_llm commit | `81001678aa901402009d5706074ccef1e6984ea0` (branch `exp/llm`, tag `S_llm-B200-2026-10-06`) |
| frozen hashes | `tilebench/llm/v2/manifests/frozen/B200_base_2026-10-06.json` (study config hash, manifests, profile file sha + seal, scoring table sha, 45 contract hashes, skill hashes, templates, execution policy) |
| campaign branch | `exp/llm-b200` cut from S_llm, worktree `../llm_b200_wt` |
| campaign identity | `formal_b200_base_2026-10-06` (Triton and cuTile x gpt and claude, Base, formal) — excluded pilot (manifests/excluded_campaigns.yaml) |
| environment | dgx003, `LD_LIBRARY_PATH` unset, bwrap sandbox, worker wall-clock 1800 s |

Campaigns run from their device branch at S_llm; the shared framework changes only through `main`. A framework
change after formal data exists requires a new, versioned campaign identity, never an in-place edit.
