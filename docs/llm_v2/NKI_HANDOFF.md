# NKI_HANDOFF — what the Trn2 environment must establish before NKI joins v2

Status at S_llm candidate: **not ready**. The shared framework carries the
NKI task identity, loader/permission checks, support-matrix entry and the
`neuron_runtime_trace` adapter slot; nothing NKI-specific has been executed
in this phase (no Neuron host here; `neuronx-cc`/`nki` are not installed in
`tilebench_env`).

## 1. What exists in main (audited, not changed)

From `docs/llm_v2/MIGRATION_AUDIT.md §1.10`:

- Real-input timing: `tilebench.core.nki_orchestrator.profile_case_on_neuron`
  → profile worker `_timed_windows(run_once, warmup, repeat)` records XLA
  execution-index windows → `tilebench.core.nki_timer.time_windows` returns
  `mean/min/max/stdev`, `repeat = len(windows)`, `per_iteration_ms`,
  `method="neuron_rt_inspect"`, `profile_input_mode="runtime_inspect_real_inputs"`.
  This is the only path that yields `nki_ms` in the manual campaign.
- Tool-default capture: `nki_timer.profile_neff` (`--profile-nth-exec`,
  `repeat=1`, `profile_input_mode="tool_default"`) is override-only and never
  becomes `nki_ms`. It must NOT be used as a v2 latency.
- Autotune candidate timing (`NkiAutotuner`, warmup 5 / iters 10, numpy or
  wall-clock) is selection-only; irrelevant to v2 (no autotuning).
- Exact-winner replay, artifact identity (NEFF sha256 matching), marker
  validation and per-spec manifests exist and are CPU-tested.

## 2. What v2 needs from the adapter (to be validated on Trn2)

1. `measure(f, warmup=1, repeat=3)` semantics: one warmup execution, then
   three timed executions of the whole `run()` (all graphs of one call summed
   per execution, as `time_windows` does), returning the three raw samples
   and their arithmetic mean. Confirm `time_windows` can return per-window
   values (it computes `per_iteration_ms`; verify it is per window, not
   averaged).
2. Timing boundary: whole operator including contract-required
   preprocessing; no host wall-clock; no compile time inside the windows.
   Confirm that the first call's compilation is excluded by the warmup and
   that `repeat` windows contain only device execution.
3. Cache policy: there is no L2 flush equivalent; record the executed policy
   explicitly in the timing record (`cache_policy: "none (Neuron runtime)"`).
4. Fresh-input / anti-cache checks (`evaluation/anticache.py`) must run on
   the XLA device: inputs moved with `to_xla_device`, outputs compared on CPU.
   Verify that `refill_in_place` (same storage, new values) is meaningful
   under XLA tensor semantics; if XLA re-materializes storage, document that
   the same-address check is replaced by a value-change check.
5. Isolation: the worker must run in a fresh process per candidate with the
   Neuron compile cache redirected into the sandbox
   (`NEURON_COMPILE_CACHE_URL` or the equivalent for the installed SDK).
6. Beta 5 (nki 0.5.0 / Neuron SDK 2.31 per the private guide's frontmatter)
   compatibility of the above; record exact `neuronx-cc`, `torch-neuronx`,
   `torch-xla`, `nki` versions in the Device Context snapshot.
7. Interference: serialize measurements with `orchestration/locks.py`
   (`device_lock("Trn2")`) and confirm no other process uses the NeuronCores
   during timing (`neuron-top`).

## 3. Eligibility registration (before any generation)

All 110 (operator, dtype) tasks are `needs_review` for Trn2
(`tasks/support.py`). On the Trn2 host run, per task, the torch reference
and input generation on the XLA device and record `eligible` /
`unsupported` with the reason, into a file that `support.py` will read
(a `trn2_eligibility.json` next to the device snapshot). Ordinary generation
failures must never change this table afterwards.

## 4. Private material and permissions

- `~/.claude/skills/nki-guide/SKILL.md` (sha256 fc04e3e1…, 159 lines) is the
  only NKI document found; it is PRIVATE. `skills/manifest.json` registers
  `reference/nki@beta5` with `permission: private`, `status: draft` and no
  in-repo path; the loader refuses to send it to a provider
  (`SkillPermissionError`).
- Before any live NKI campaign the owner must confirm: (a) the licence of
  that document, (b) whether a derived Reference Skill may be sent to
  OpenAI/Anthropic, (c) whether generated NKI kernels and distilled NKI
  skills may be stored outside the private repository. Until then the
  Trn2 preflight blocks with "NKI timing adapter is not validated".

## 5. Steps on the Trn2 host (one-time, with evidence)

```
python -m tilebench.llm.v2 doctor                      # versions, device, manifests
python -m tilebench.llm.v2 capture-device              # fill skills/device/Trn2/<snapshot>/SKILL.md unknowns
python -m pytest tests/llm_v2 -q                       # CPU tests must pass there too
# adapter validation (to be written in evaluation/nki_adapter.py on exp/llm):
#   - 1 warmup + 3 timed windows on a manual operator with real inputs
#   - compare the three samples with the manual campaign's method
#   - record the timing record JSON as evidence under outputs/llm_v2/preflight/Trn2/
python -m tilebench.llm.v2 preflight --devices Trn2
```

The adapter implementation goes back to `exp/llm` for review; the device
branch `exp/llm-trn2` only holds device configuration and run evidence.
