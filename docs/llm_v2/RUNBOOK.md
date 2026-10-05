# RUNBOOK — TileBench++ LLM protocol v2 (shared framework phase)

All commands run from a checkout root with the `tilebench_env` (or the
device's equivalent) environment active. Nothing below calls a paid API
unless stated; the live campaign commands are gates that refuse to start on
any blocker.

## Daily checks (any host, no GPU needed)

```
python -m tilebench.llm.v2 doctor                 # versions, device, manifests, skills, contracts
python -m tilebench.llm.v2 validate-manifests     # study/folds/modes/skills/contracts; exit 1 on errors
python -m tilebench.llm.v2 inventory              # registered assets with injected hashes
python -m pytest tests/llm_v2 -q                  # CPU/mock tests of the protocol
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
python -m tilebench.llm.v2 dry-run --device MI300X --dsl triton --allow-draft
```

`--allow-draft` accepts draft skills/contracts for development; live
campaigns require approved contracts and approved/frozen skills.

## Mock execution (ten rounds, persistence, resume)

```
python -m tilebench.llm.v2 run-mock --out outputs/llm_v2/mock --operator vector_add --dtype fp16 --device B200 --dsl triton
python -m tilebench.llm.v2 run-mock --out outputs/llm_v2/mock --operator vector_add --dtype fp16 --device B200 --dsl triton --resume
python -m tilebench.llm.v2 coverage outputs/llm_v2/mock
python -m tilebench.llm.v2 metrics outputs/llm_v2/mock --budgets 1000 5000 20000
python -m tilebench.llm.v2 review-queue outputs/llm_v2/mock
```

## Preflight and campaign gates

```
python -m tilebench.llm.v2 preflight                                # every device/DSL/condition, development gates
python -m tilebench.llm.v2 preflight --devices B200 GH200 --live    # live gates (model ids, approved assets, frozen folds)
python -m tilebench.llm.v2 base --device B200 --dsl triton          # refuses on blockers; no API call in this phase
python -m tilebench.llm.v2 enhanced --device GH200 --dsl cutile     # idem; additionally needs frozen fold skills
python -m tilebench.llm.v2 distill --dsl triton --fold A --index outputs/llm_v2/base_index.json
python -m tilebench.llm.v2 distill --dsl triton --mode release --index outputs/llm_v2/base_index.json
```

The live runner invocation (provider + `SubprocessEvaluator` per device) is
wired on the device branches after `S_llm` is frozen; the shared branch
ships the gate only, so no one can start a paid campaign from it by
accident.

## Skills and device snapshots

```
python -m tilebench.llm.v2 skills-register --kind reference --key triton --version 3.6.0 \
    --path skills/reference/triton/3.6.0/SKILL.md --permission public --status approved --source "<owner note>"
python -m tilebench.llm.v2 capture-device > outputs/llm_v2/device_capture_$(hostname).json
```

Re-registering an asset recomputes its hashes; approving one is the
owner's explicit act (status `approved`/`frozen`), never automatic.

## Review bundle

```
python -m tilebench.llm.v2 export-review-bundle --out outputs/llm_v2/review_bundle
```

Copies docs, contracts, templates, manifests and public/internal skill
texts with sha256s; excludes credentials, private bodies and
`outputs/`.

## Remote devices (GH200 / MI300X / Trn2) — one-time evidence

On each device host, from the device branch cut at `S_llm`:

```
python -m tilebench.llm.v2 doctor
python -m tilebench.llm.v2 capture-device          # fills the device snapshot's unknowns (see skills/device/CAPTURE_CHECKLIST.md)
python -m pytest tests/llm_v2 -q
python -m tilebench.llm.v2 dry-run --device <DEV> --dsl <DSL> --allow-draft --limit 3
python -m tilebench.llm.v2 preflight --devices <DEV> --live
```

Trn2 additionally follows `docs/llm_v2/NKI_HANDOFF.md`.
