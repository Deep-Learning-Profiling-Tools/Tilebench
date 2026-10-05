# PROMPT_SPEC — TileBench++ LLM protocol v2

Templates live in `tilebench/llm/v2/prompts/templates/*.md`; the renderer is
`tilebench/llm/v2/prompts/renderer.py` (placeholders `{{name}}`, a missing
value is an error). Rendered snapshots produced by the actual renderer are
under `docs/llm_v2/prompt_snapshots/` (see `make_snapshots` in
`tilebench/llm/v2/devtools.py`); synthetic components used there are marked
TEST-ONLY and are never referenced by `skills/manifest.json`.

## 1. Request shape

One API request = one system message + one user message. One request
produces one DSL implementation file for one task
`(operator, dtype, case_id, device, dsl, model, condition)`. Nothing from
another task, DSL, model, device, dtype or condition is ever rendered.

| Template | Used for | Rendered by |
|---|---|---|
| `system_interface.md` | every generation request | `render_system` |
| `initial.md` | round 1 (also round 1 of every Enhanced trajectory) | `render_initial` |
| `refinement.md` | rounds 2..10, first generation of the round | `render_refinement` |
| `compliance_repair.md` | same-round regeneration after a confirmed violation (attempt 2, 3) | `render_repair` |
| `dev_contract_extraction.md` | development only: contract extractor (reads manual code) | `render_dev_extraction` |
| `dev_contract_reconciliation.md` | development only: reconciliation of two extractions | `render_dev_reconciliation` |
| `distill_evidence_extraction.md` | distillation map step (one trajectory) | `render_distill_extraction` |
| `distill_synthesis.md` | distillation reduce step | `render_distill_synthesis` |
| `reviewer.md` | optional LLM compliance reviewer (disabled by default) | `render_reviewer` |

The development extractor and the kernel generator never share a session,
a template, or a provider object; the extractor templates are only used by
the development CLI and are not reachable from `orchestration.runner`.

## 2. Fixed system/interface rules

`system_interface.md` states, verbatim in English: implement one fixed task
in the requested DSL on the declared device; preserve the canonical
algorithm contract while optimizing permitted implementation details; use
the supplied version-pinned API reference and device context; commit to one
deterministic implementation and configuration: tunable values (tile/block
sizes, warps, stages, vector widths) are fixed literals, per stage for
multi-kernel implementations, while quantities derived deterministically
from the fixed shape and those literals (grid sizes, loop bounds, chunk
counts) may be computed, a later round may choose another configuration,
and `get_last_config()` returns the same dict on every call; do not invoke
autotuners, search or time several configurations at runtime, reuse cached
outputs, delegate the computation to the reference or external compute
libraries, or inspect/modify the evaluator; PyTorch only for the operations
the contract permits; correctness is checked on freshly generated inputs
(fresh allocation, then new values in the same storage, then a repeated
call on identical inputs) and timing uses one fixed input set, no state
across calls, no input mutation unless the contract declares it; return
exactly one fenced block titled with the output file, `run` called
positionally with the task inputs only (no `block_size`/`autotune`
keywords); performance feedback is the GPU time of the kernels launched by
`run()` (one warmup, mean of three timed launches; host-side work is not
GPU time, every launched kernel/fill/copy/cast is counted).

The 45 contracts state the same call form ("called positionally with
exactly the inputs listed; no keyword arguments are passed"); the legacy
`block_size=...`/`autotune=False` call descriptions were removed on
2026-10-05.

## 3. Components of the user message (order fixed)

1. `# API reference: <dsl> <version>` — Reference Skill text (manifest-pinned).
2. `# Device context: <device>` — Device Context Skill text.
3. `# Optimization guidance ...` — Enhanced only; the frozen fold skill.
4. `# Canonical algorithm contract: <op>` — `contract.md` of the operator.
5. `# Task` — operator, DSL/version, output file, dtype (+ FP8 format),
   the fixed case parameters (from the selected real case), the effective
   numerical tolerance and its source, the `run` interface and return
   structure.
6. PyTorch functional reference — `impl_torch.py` with non-functional
   top-level definitions removed (`tasks.fields.functional_reference`,
   mapping recorded in `dropped_definitions`).
7. Output requirements.

Base and Enhanced differ only in component 3. Every component's injected
hash is recorded in `trajectory.json.content_hashes`.

Never rendered: `config.yaml` as a whole, `metrics:` (flops/bytes
expressions), `benchmark:` settings, peak numbers, manual kernel sources,
manual latencies, autotune winners.

## 4. Feedback whitelist (refinement)

`prompts/feedback.py` is the only path from evaluator output to a prompt.
Allowed: round status text, `latency_ms_mean` and the three
`latency_ms_samples` of VALID rounds, the candidate's own `get_last_config`
literal, sanitized diagnostics (max 60 lines / 6000 chars, fixed rule; lines
containing roofline/T_SOL/efficiency/speedup/stop_score/pct_peak/human/
torch_ms/score are replaced by `[line withheld]`). Forbidden fields are
listed in `study.yaml:feedback.forbidden_fields`; `assert_feedback_clean`
is applied in tests.

History rule (`study.yaml:trajectory.history_rule`): previous round's
candidate + outcome; best valid candidate when different; runtime table of
all valid rounds. Regression: the slower valid candidate is recorded, the
best is kept. Context overflow: `prompt_too_long` before any API call; no
truncation; the attempt is recorded with cost 0 / `not_sent`.

Rounds without a compliant candidate (`renderer._prev_block`): when the
previous round closed with three contract violations, or without a
parsable file, the refinement prompt shows the outcome and diagnostics
only, states that rejected candidates are not shown and must not serve as
the basis of the next implementation, and offers the trajectory's last
compliant implementation (or says none exists). A violating candidate is
never shown as the previous candidate. Round statuses that can appear in
the outcome text: valid, format_error, interface_error (missing/failing
`run`/`get_last_config`, or a configuration that changed between calls),
compile_error, runtime_error, numerical_error, timing_error,
contract_violation, infrastructure_incomplete.

## 5. Compliance repair

Rendered only after a CONFIRMED violation: static evidence level
`confirmed` (autotune entry points resolved through the file's import
aliases, host-scope reference-library calls, forbidden imports), a contract
rule of level `confirmed` matched in its declared scope on a computational
line of code-only text, or a violation confirmed by the evaluator at
execution (autotuner object in the module, output aliasing an input).
States the round, the attempt number and the maximum (3), lists the
violations with line numbers, shows the rejected candidate, and offers the
trajectory's last compliant implementation as the fallback (never the
violating one). Ordinary compile/numerical/interface failures never reach
this template. Suspicious evidence (`review_required`) blocks the
trajectory for a recorded human decision (`review-resolve`); it never
triggers a repair by itself.

## 6. Distillation prompts

Both state the fixed scope (DSL/version, source device, training folds,
held-out fold, base condition only). The map step extracts conditional
observations with applicability, effect, confounders, supporting and
contradicting round ids and limits; the reduce step synthesizes rules each
citing supporting/contradicting trajectory ids, with no operator-specific
solutions, no winner recipes, no claims about other devices. Access to
trajectories is enforced in `distillation/access.py`, not by the prompt.

## 7. Parser

`validation/parser.py`: exactly one fenced `python` block with
`title="<output_file>"`; it must parse as Python. Anything else is a
`format_error` (consumes the round, no repair). A response the provider
reports as truncated (`incomplete:max_output_tokens`, `max_tokens`) is a
`format_error` as well, even if its text happens to parse: it is not a
complete candidate.

## 8. Providers

Requests are streamed (`client.responses.stream`,
`client.messages.stream`) and consumed to the end; the archived result
carries the echoed model id, response id, terminal status, truncation flag,
event count, raw usage and the parameters actually sent (never the key).
Configuration errors (400/401/403/404/422) are never retried; transport
failures are retried up to `models.yaml:transport.max_transport_retries`
with each attempt logged and classified as `charged: no` (never reached the
provider or rejected with an error status) or `charged: unknown` (timeout
after sending, interrupted stream; the last streamed usage is kept as a
lower bound).
