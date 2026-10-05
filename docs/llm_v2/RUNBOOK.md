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
python -m pytest tests/llm_v2 -q                  # CPU/mock tests of the protocol (123 tests)
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

## Mock execution (ten rounds, persistence, resume; no API, no GPU)

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
are reused, never re-requested; the config hash, content hashes, run type
and generator settings must be unchanged); `--stop-after-rounds N` pauses
after N closed rounds (state is resumable); `--isolation auto|bwrap|none`
(auto = bubblewrap when it works, recorded in every result);
`--worker-timeout`, `--max-trajectories`, `--out-root` (default
`outputs/llm_v2`).

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

## Compliance review (human decision)

A `review_required` verdict (suspicious static evidence) blocks the
trajectory. Inspect and decide:

```
python -m tilebench.llm.v2 review-resolve --trajectory-dir <dir> --show
python -m tilebench.llm.v2 review-resolve --trajectory-dir <dir> --decision compliant --note "<evidence>" --reviewer <name>
python -m tilebench.llm.v2 review-resolve --trajectory-dir <dir> --decision violation --note "<evidence>" --reviewer <name>
python -m tilebench.llm.v2 base ... --resume        # continue
```

Decisions are appended to `reviews.jsonl` and to `trajectory.json.notes`.
There is no automatic clearance; the optional LLM reviewer is disabled.

## Distillation (**paid** in formal mode; refuses without its gates)

```
python -m tilebench.llm.v2 distill --campaign-dir outputs/llm_v2/<base_campaign> --dsl triton --fold A --out outputs/llm_v2/distill/triton_A
python -m tilebench.llm.v2 distill --campaign-dir ... --dsl triton --mode release --out ...
python -m tilebench.llm.v2 distill --campaign-dir ... --dsl triton --fold A --run-type test-only --out ...   # test-only skill, never registered
```

Formal mode needs an approved distiller model, frozen folds and formal
Base trajectories of the source device and training folds (the trusted
index is built from the campaign directory with file hashes; identity,
root containment and hashes are re-verified before any state is read).
Observations and the synthesis are persisted under `--out` and reused on
resume. Registration of a produced skill in `skills/manifest.json` is a
separate, explicit owner act (`skills-register`).

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
