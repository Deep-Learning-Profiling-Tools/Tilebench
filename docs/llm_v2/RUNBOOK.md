# RUNBOOK — TileBench++ LLM protocol v2

All commands run from a checkout root with the `tilebench_env` (or the
device's equivalent) environment active. Commands marked **paid** send real
requests to the configured providers; every other command is local.

Keys: the adapters read `OPENAI_API_KEY` and `CLAUDE_API_KEY` from the
environment (`manifests/models.yaml:api_key_env`). They are never written
to a file; the evaluation worker runs with every `*KEY*`/`*TOKEN*`/
`*SECRET*` variable removed and, under bubblewrap, with the user's home
hidden.

## Daily checks (any host, no GPU needed)

```
python -m tilebench.llm.v2 doctor                 # versions, device, manifests, skills, contracts, isolation backend
python -m tilebench.llm.v2 validate-manifests     # study/folds/modes/skills/contracts/rule scopes; exit 1 on errors
python -m tilebench.llm.v2 inventory              # registered assets with hashes and provider/publication grants
python -m pytest tests/llm_v2 -q                  # CPU/mock tests of the protocol (155 tests)
```

## Task definition

```
python -m tilebench.llm.v2 select-cases --out outputs/llm_v2/cases.json   # representative case per (op, dtype)
python -m tilebench.llm.v2 tasks --out outputs/llm_v2/tasks.json          # eligibility table (880 rows)
```

## Prompts

```
python -m tilebench.llm.v2 render --operator vector_add --dtype fp16 --device B200 --dsl triton --allow-draft --out outputs/llm_v2/render/vector_add
python -m tilebench.llm.v2 dry-run --device B200 --dsl triton --allow-draft --limit 5       # every eligible task, no API call
python -m tilebench.llm.v2 snapshots --out docs/llm_v2/prompt_snapshots/B200_triton_vector_add_fp16
```

`--allow-draft` accepts draft skills/contracts for development; formal
campaigns require approved contracts and approved/frozen skills.

## Mock execution (five rounds, one generation each, persistence, resume; no API, no GPU)

```
python -m tilebench.llm.v2 run-mock --out outputs/llm_v2/mock --operator vector_add --dtype fp16 --device B200 --dsl triton
python -m tilebench.llm.v2 run-mock --out outputs/llm_v2/mock --operator vector_add --dtype fp16 --device B200 --dsl triton --resume
python -m tilebench.llm.v2 coverage outputs/llm_v2/mock
python -m tilebench.llm.v2 metrics outputs/llm_v2/mock --budgets 1000 5000 20000
python -m tilebench.llm.v2 review-queue outputs/llm_v2/mock
```

## Provider probe (**paid**, one short request)

```
python -m tilebench.llm.v2 probe-provider --model gpt    --out outputs/llm_v2/probes/gpt_probe.json
python -m tilebench.llm.v2 probe-provider --model claude --out outputs/llm_v2/probes/claude_probe.json
```

Uses exactly the configured model id, effort and output cap (never a
smaller model or a lower cap); records the echoed model id, response id,
terminal status, stream event count and raw usage. A refusal (auth,
parameter, model) is reported verbatim with exit code 2.

## Preflight

```
python -m tilebench.llm.v2 preflight                                            # development report, every device/DSL/condition
python -m tilebench.llm.v2 preflight --devices B200 --live                      # formal gates (approved contracts/skills/models, frozen folds)
python -m tilebench.llm.v2 preflight --devices B200 --live --run-type validation --provider openai
```

Gates by run type:

| gate | formal | validation |
|---|---|---|
| contracts | all approved | present (draft/needs-review recorded as warning) |
| skills | approved/frozen | draft or better |
| provider grant (`sendable_to`) | required | required |
| models.yaml status | approved | candidate or approved |
| folds (Enhanced) | frozen | frozen |
| timing adapter ready, host arch = campaign device | required | required |

## Live campaigns (**paid**; GPU; device lock)

```
python -m tilebench.llm.v2 base --device B200 --dsl triton --model gpt --campaign <name> --run-type validation \
    --operators vector_add softmax histogramming --dtypes fp16 int32
python -m tilebench.llm.v2 base --device B200 --dsl cutile --model claude --campaign <name> --run-type validation --operators vector_add
python -m tilebench.llm.v2 enhanced --device B200 --dsl triton --model gpt --campaign <name>         # formal: needs frozen fold skills
```

Options: `--resume` continues persisted trajectories (archived responses
are reused, never re-requested; the config hash, content hashes, run type,
generator settings AND the evaluator fingerprint must be unchanged — a
formal resume refuses any evaluator change, a validation resume accepts one
only with `--allow-evaluator-change` and records it in
`evaluator_changes.jsonl`); `--retry-incomplete` re-evaluates a round
closed by an evaluation-side infrastructure failure (new `eval_NNNN`
revision, no request); `--resume-transport --reason "<why>"` re-opens an
attempt closed by exhausted transport retries or a provider refusal (the
transport history is rebuilt from `transport.jsonl`, unknown charges keep
propagating, a `reopened` event is recorded, no round is added);
`--stop-after-rounds N` pauses after N closed rounds; `--isolation
auto|bwrap|none` (formal runs require bwrap and a passing isolation probe);
`--worker-timeout`, `--executor`, `--max-trajectories`, `--out-root`
(default `outputs/llm_v2`).

Evidence is append-only: the first `compliance.json` of an attempt is never
rewritten (rechecks are `compliance_recheck_NNNN.json`), every evaluation is
an `eval_NNNN/` revision with `META.json` (reason, supersedes, executor,
evaluator fingerprint, candidate sha256).

Independent re-evaluation of an archived candidate (no request):

```
python -m tilebench.llm.v2 re-evaluate --trajectory-dir <dir> --round 1 --reason "<why>" --executor "<who>" [--adopt]
```

Chain per task: manifests → eligibility → task context (prompt) +
evaluation job (tolerance, rules, timing; asserted equal to the prompt's
tolerance) → provider (factory from models.yaml) → `TrajectoryRunner`
(render → stream request → parse → compliance → evaluate) →
`SubprocessEvaluator` (sandbox, device lock, worker) → trajectory.json /
usage.jsonl / transport.jsonl / per-attempt artifacts.

Several processes may run concurrently (different models/DSLs/operators):
API calls overlap, GPU evaluations serialize on `outputs/llm_v2/locks/<device>.lock`.

Monitoring a campaign directory:

```
python -m tilebench.llm.v2 coverage      outputs/llm_v2/<name>
python -m tilebench.llm.v2 review-queue  outputs/llm_v2/<name>
python -m tilebench.llm.v2 metrics       outputs/llm_v2/<name>       # validation runs are labelled unscored
```

## Compliance review (revision 4: the supervising Claude Code session adjudicates)

A `review_required` verdict (suspicious static evidence) blocks the trajectory. The supervising Claude Code
session polls the queue and decides from the candidate, the frozen contract, the evaluator rules, the exact
checker evidence and the interface only (never latency, speedup, T_emp, E(B), cost or ranking); no reviewer API
call, no human dependency (`study.yaml compliance.adjudicator: claude-code`):

```
python -m tilebench.llm.v2 review-queue outputs/llm_v2/<campaign>
python -m tilebench.llm.v2 review-packet --trajectory-dir <dir>          # evidence; asserts no performance/cost field
python -m tilebench.llm.v2 review-resolve --trajectory-dir <dir> --decision compliant --note "<rationale>"
python -m tilebench.llm.v2 review-resolve --trajectory-dir <dir> --decision violation --note "<rationale>"
```

The running `schedule` process polls every blocked trajectory (30 s) and continues it in the same process (same
provider limiters, same global evaluation budget) as soon as its decision is recorded; while reviews are pending
and nothing else runs, the process waits for them (STOP ends the wait). Decisions are appended to the round's
`reviews`, to `reviews.jsonl` and to `trajectory.json.notes` with candidate/contract/rules/checker hashes.
A checker or contract bug found during review pauses the campaign (STOP) and is versioned; never rewrite a
frozen contract mid-campaign (review-resolve refuses a changed contract hash).

## Empirical calibration (GPU; exclusive with campaigns; writes only a new calibration directory)

```
python -m tilebench.llm.v2 calibrate --device B200                       # dgx003: run with LD_LIBRARY_PATH unset
python -m tilebench.llm.v2 calibration-check artifacts/llm_v2/calibration/B200/<id>/profile.json
python -m tilebench.llm.v2 calibration-register --device B200 --profile artifacts/llm_v2/calibration/B200/<id>/profile.json --status candidate
python -m tilebench.llm.v2 calibration-register --device B200 --profile ... --status frozen --by "<owner>"   # owner decision
python -m tilebench.llm.v2 scoring-table --device B200 --out artifacts/llm_v2/scoring/B200/<file>.json
```

`calibrate --quick` smoke-tests the code path (never registrable). The
legacy table `tilebench/data/peak_performance/<GPU>.json` is never written;
`scripts/measure_peak.py` is not part of the v2 chain. See
`docs/llm_v2/CALIBRATION.md` for the protocol, the modes and the per-device
instructions (GH200 / MI300X run the same command locally; Trn2 needs the
native adapter from the NKI window).

## Formal B200 Base (frozen configuration, 2026-10-06)

Run from the `exp/llm-b200` worktree (cut from `S_llm`), with the campaign
environment: `LD_LIBRARY_PATH` unset (the dgx003 default mixes the wheel
cuBLAS with a system cuBLASLt; formal preflight fails closed on that),
`OPENAI_API_KEY` / `CLAUDE_API_KEY` present (never printed).

```
env -u LD_LIBRARY_PATH python -m tilebench.llm.v2 preflight --devices B200 --live --run-type formal
env -u LD_LIBRARY_PATH python -m tilebench.llm.v2 base --device B200 --dsl triton --model gpt    --campaign <id> --run-type formal --operators vector_add --dtypes fp16   # launch acceptance
env -u LD_LIBRARY_PATH python -m tilebench.llm.v2 base --device B200 --dsl triton --model gpt    --campaign <id> --run-type formal --resume   # full task set, same identity
```

Formal gates (all must pass): frozen + approved folds (Base and Enhanced),
45 approved contracts, approved Reference Skills and the pinned Device
Context (`study.yaml device_snapshots`), approved arithmetic declaration
with every task `ok`, frozen empirical profile (file sha + seal), scoring
binding, consistent BLAS stack, bwrap probe, host arch, approved
generators, worker timeout = `study.yaml evaluation.worker_timeout_s`
(1800 s; any other value is refused for formal runs), capture failures are
`timing_error` (consumed round). A formal `review_required` round pauses
until `review-resolve --reviewer <designated human> --note <evidence>`
records the decision (candidate sha256, checker and rules hashes are
written to `reviews.jsonl`).

## Clean-tree publication check

```
T=$(mktemp -d); git archive HEAD artifacts/llm_v2/<name> | tar -x -C $T   # then verify INDEX.json hashes under $T
python -m pytest tests/llm_v2/test_freeze_fixes.py -k clean_git_archive -q
```

## Distillation (**paid** in formal mode; refuses without its gates)

```
python -m tilebench.llm.v2 distill --campaign-dir outputs/llm_v2/<base_campaign> --dsl triton --fold A --out outputs/llm_v2/distill/triton_A
python -m tilebench.llm.v2 distill --campaign-dir ... --dsl triton --mode release --out ...
python -m tilebench.llm.v2 distill --campaign-dir ... --dsl triton --fold A --run-type test-only --out ...   # test-only skill, never registered
```

Formal mode needs an approved distiller model, frozen folds and formal,
COMPLETE Base trajectories of the source device and training folds (the
trusted index is built from the campaign directory with file hashes and
the frozen fold manifest — folds are recomputed per operator, a state whose
own label disagrees is excluded; identity, root containment and hashes are
re-verified before any state is read; coverage against the pre-declared
task set is required unless `--allow-partial-coverage` is recorded).
Materials per trajectory: every attempt's source, verdict, diagnostics and
diff, every round's raw samples, plus the offline SOL record
(`sol_info.json`). Observations and the synthesis are persisted with the
full request/raw response under `--out` and reused on resume by input
identity; a truncated or interrupted response is kept as a partial record
with its cost and never becomes an observation or a skill (exit 3).
Registration of a produced skill in `skills/manifest.json` is a separate,
explicit owner act (`skills-register`); the Enhanced loader validates the
skill's own provenance manifest (DSL, version, source device, mode, folds,
source ids, content hash) before injecting it.

## Skills and device snapshots

```
python -m tilebench.llm.v2 skills-register --kind reference --key triton --version 3.6.0 \
    --path skills/reference/triton/3.6.0/SKILL.md --permission public --status approved \
    --sendable-to openai anthropic --publishable --source "<owner note>"
python -m tilebench.llm.v2 capture-device > outputs/llm_v2/device_capture_$(hostname).json
```

Three independent decisions per asset: `status` (content approval),
`sendable_to` (per provider; an empty list is not a grant), `publishable`
(may leave the private repository). `private` assets can carry none.

## Publication export (Git-tracked copy of a campaign)

```
python -m tilebench.llm.v2 export-publication --campaign-dir outputs/llm_v2/<name> --out artifacts/llm_v2/<name>
```

Copies every round's request/response/candidate/compliance/evaluation
(and the archived worker evidence), the ledgers and the state, with an
`INDEX.json` of sha256s; withholds any trajectory whose context includes a
non-publishable component (`REDACTIONS.json`). See
`docs/llm_v2/ARTIFACT_POLICY.md`.

## Review bundle

```
python -m tilebench.llm.v2 export-review-bundle --out outputs/llm_v2/review_bundle
```

Copies docs, contracts, templates, manifests and publishable skill texts
with sha256s; excludes credentials, private / non-publishable bodies and
`outputs/`.

## Remote devices (GH200 / MI300X / Trn2) — one-time evidence

On each device host, from the device branch:

```
python -m tilebench.llm.v2 doctor
python -m tilebench.llm.v2 capture-device          # fills the device snapshot's unknowns (see skills/device/CAPTURE_CHECKLIST.md)
python -m pytest tests/llm_v2 -q
python -m tilebench.llm.v2 dry-run --device <DEV> --dsl <DSL> --allow-draft --limit 3
python -m tilebench.llm.v2 preflight --devices <DEV> --live --run-type validation --provider openai
python -m tilebench.llm.v2 base --device <DEV> --dsl <DSL> --model gpt --campaign <name> --run-type validation --operators vector_add --dtypes fp16
```

The same `base`/`enhanced` commands run there; nothing is re-implemented
per device. Trn2 additionally follows `docs/llm_v2/NKI_HANDOFF.md` (its
timing adapter is not ready; the preflight blocks it).

## Revision-4 formal campaign (one process, both models, 20 cases per candidate)

```
# all models and DSL tracks in ONE process; hard global evaluation backlog; LD_LIBRARY_PATH unset on dgx003
env -u LD_LIBRARY_PATH nohup python -m tilebench.llm.v2 schedule --device B200 --models gpt claude --campaign <name> \
    --dsls triton cutile tilelang --run-type formal --resume --max-pending-evaluations 16 --executor <who> \
    > outputs/llm_v2/campaign_logs/<name>.log 2>&1 &
touch outputs/llm_v2/<name>/STOP                 # graceful pause (remove to resume)
python -m tilebench.llm.v2 review-queue outputs/llm_v2/<name>
python -m tilebench.llm.v2 cost-report outputs/llm_v2/<name>
python -m tilebench.llm.v2 metrics outputs/llm_v2/<name>      # geomean efficiency over the 20 cases; speedup vs stored torch
```

## Revision-3 formal campaign (schedule, cost, pause) — superseded pilot

```
# one process per model; all three DSL tracks; adaptive concurrency; LD_LIBRARY_PATH unset on dgx003
env -u LD_LIBRARY_PATH nohup python -m tilebench.llm.v2 schedule --device B200 --model gpt    --campaign <name> \
    --dsls triton cutile tilelang --run-type formal --resume --executor <who> > outputs/llm_v2/campaign_logs/<name>_gpt.log 2>&1 &
env -u LD_LIBRARY_PATH nohup python -m tilebench.llm.v2 schedule --device B200 --model claude --campaign <name> \
    --dsls triton cutile tilelang --run-type formal --resume --executor <who> > outputs/llm_v2/campaign_logs/<name>_claude.log 2>&1 &
touch outputs/llm_v2/<name>/STOP                 # graceful pause (remove to resume)
python -m tilebench.llm.v2 cost-report outputs/llm_v2/<name>
python -m tilebench.llm.v2 metrics outputs/llm_v2/<name>      # E_token and E_usd, 45-operator denominator
```
