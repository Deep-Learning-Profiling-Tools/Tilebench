# LLM track v2: migration audit of the legacy pipeline (S_main)

Worktree `exp/llm` at `ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`. Every `path:line` below was read in this tree. Statements marked (inference) are not literally in the code.

## 1. Verified migration points

### 1.1 generate.py: joint Triton/cuTile generation, no early-stop
- One response per iteration is parsed into two files: `files = parse_response(response_text)` (`tilebench/llm/generate.py:258`); a missing `impl_<b>.py` is a `fatal` feedback (`:259-265`); otherwise each is written to `iter_dir/impl_<b>.py` (`:266-267`). `parse_response` (`tilebench/llm/prompt_builder.py:540-555`) matches fenced blocks with `_CODEBLOCK_RE` (`:534-537`) and normalises titles so `impl_triton.py`, `triton`, `impl-triton` all map to `impl_triton.py` (`:550-554`).
- Parsing only runs on a cache miss: `cache_complete = response_path.exists() and all(p.exists() ...)` (`generate.py:227`) reuses `response.md` and the existing impl files (`:228-230`).
- Loop control: `for i in range(args.max_iters): active_backends = BACKENDS  # both backends generated every iter` (`generate.py:325-326`), with `BACKENDS = ("triton", "cutile")` (`:45`). The only exit is budget exhaustion (`:402-406`). Comments confirm: "There is no roofline-based early stopping" (`:320-321`, docstring `:15-16`).
- Verdict: there is NO early-stop in the current loop. "Remove existing early-stop" is NOT a pending task; `stop_score_*` is a legacy field name only (`tilebench/llm/evaluator_runner.py:459-469`).

### 1.2 prompt_builder.py: skills and config injection
- `_SKILL_PATHS = {"triton": REPO_ROOT / "skills" / "triton-guide" / "SKILL.md", "cutile": REPO_ROOT / "skills" / "cutile-guide" / "SKILL.md"}` (`prompt_builder.py:61-64`); Triton/cuTile only.
- `_read_skill`: `if path is None or not path.exists(): return ""` (`:73-74`); frontmatter stripped by `raw.find("\n---\n", 4)` (`:76-79`). `_backend_ref_sections` returns `[]` when the skill is empty, "so the prompt simply omits the section" (`:124-129`). `skills/` lives at the repo root, not in the package (verified: `skills/` exists, `tilebench/skills` does not); an install without the checkout silently loses both API references (inference).
- Whole config injected: `config = _read(op_dir / "config.yaml")` (`:160`) and `"```yaml", config, "```"` (`:193-195`), repeated in the feedback prompt (`:243`, `:345-347`). This is the UNTRIMMED operator config (`operator_dir(op)`), while the evaluator runs the trimmed copy (1.4); the prompt tells the LLM "Your kernels must work for all combinations" (`:197`).

### 1.3 evaluator_runner.py: timing protocol
- `_proton_time_rotating(..., warmup: int = 5, repeat: int = 20, flush_l2: bool = True, scope_name: str = "launch")` (`evaluator_runner.py:52-60`); both call sites pass `warmup=5, repeat=20, flush_l2=True` explicitly (`:267-270` kernel, `:273-276` torch).
- `peak = load_peak("B200")` (`:351`); `def load_peak(gpu_name: str = "B200")` (`tilebench/llm/roofline.py:20`).
- Core timer now has `DEFAULT_WARMUP = 1`, `DEFAULT_REPEAT = 3` (`tilebench/core/timer.py:43-44`; engine reads them at `tilebench/core/engine.py:129-130`). The evaluator imports only `_build_profile_base, _find_scope_mean_ns, _flush_l2_cache, _load_profile_data` (`evaluator_runner.py:41-43`) and never reads the config's `benchmark:` block (only `config.get("metrics", {})` `:352` and `expand_cases` `:333`).
- Timing helper: pre-generates `warmup + repeat` input tuples (`:71-75`, 25 tuples alive at once); `_flush_l2_cache()` before every warmup (`:79-81`) and every timed launch (`:89-90`); `proton.start(..., context="shadow", data="tree", backend=None)` (`:86`); `with proton.scope(scope_name): impl_run(*inp)` (`:91-92`); mean via `_find_scope_mean_ns(root, scope_name, repeat)` (`:106`). No CUDA graph (`_prepare_runner`/`use_cuda_graph` never referenced).
- Implicit runs before timing, per (backend, case) in `_run_one_case` (`:206-303`): torch reference (`:217`), verify run (`:228`), `get_last_config()` (`:230`), anti-cache: fresh inputs + torch + impl (`:247-254`). Then timing: 25 generator calls + 5 warmup + 20 timed for the kernel, and again for torch. Totals: 52 generator calls, 27 `impl.run`, 27 `impl_torch.run`. SIGALRM cap wraps verify+timing (`:222-223`, default 900 s `:324`).

### 1.4 Case trimming
- `_trim_case_grid_to_largest(cfg_text)` (`generate.py:59-143`) rewrites the YAML text of the `case_grid:` block (`:93-103`): explicit int lists become `[max]` (`:106-117`), `expr:` entries are `eval(expr_body, {"range": range})` then `max(values)` (`:121-139`); `dtype` is excluded (`:107`, `:123`). `_copy_framework_files` writes the trimmed copy into the iter dir (`:146-156`); the evaluator expands that copy with `expand_cases` (`evaluator_runner.py:332-333`), i.e. the cartesian product of per-dimension maxima (`tilebench/data/tensors.py:26-39`).
- Verdict: maxima are taken per dimension independently and re-combined; no case is selected from the operator's own `expand_cases()` list (the engine's selector is `case_indices`, `engine.py:148-155`). `test_cases`/`case_preset` configs and `case_defaults` are not trimmed (`generate.py:76-77`, `:93`). The docstring claims parity with the NCU sweep's `default_params_per_dtype` (`:66-68`) but nothing reads that catalogue.

### 1.5 Feedback content
- `_format_feedback` (`prompt_builder.py:456-527`) renders: compile tracebacks (`:463-470`), timeout text (`:472-483`), up to 10 verify failures with `params` and the assert_close message (`:485-493`), per backend "mean roofline utilization = **{stop_score}%** (report_mean=...)" (`:496-502`), per combo "`{pct}% of roofline ({bound_by}), {speedup}× torch, cfg={cfg}`" (`:504-520`), and an optimisation hint (`:521-525`).
- `_format_trajectory` (`:374-453`) adds a per-iteration table of `cfg`, `score` (stop_score), `speedup_vs_torch`, verify status (`:391-414`), defines speedup as `torch_latency / kernel_latency` (`:416-421`), and best-so-far / regression deltas in pp (`:424-452`). Best-so-far source code is re-shown with its `stop_score` (`:296-319`).
- Leaked scoring/human fields: `stop_score_<b>`, `report_arith_mean_<b>`, `roofline_pct`, `bound_by`, `speedup_vs_torch` (ratio of the two measured latencies), `cfg`, verify counts/messages, compile/timeout text, best-so-far `stop_score` and code. Absolute `latency_ms`/`torch_latency_ms` are in `feedback.json` (`evaluator_runner.py:408-417`) but NOT rendered in the prompt (`prompt_builder.py:514-517`), contrary to `system_prompt.md:9-10` and `framework_guide.md:36-37`.

### 1.6 Verifier tolerances
- Legacy: `ok, err = verify(output, ref)` (`evaluator_runner.py:231`) and `verify(fresh_out, fresh_ref)` (`:254`), no `atol/rtol`; `config_tolerance`, `arch_overrides`, `detect_arch` are absent from the file. Tolerances are therefore the per-dtype defaults (`tilebench/core/verifier.py:29-31`), while the prompt promises `verify:` tolerances (`prompt_builder.py:203`).
- Engine: `verify_atol, verify_rtol = config_tolerance(config.get("verify", {}), detect_arch())` (`engine.py:133`), passed at `:259`, `:291`, `:323`.

### 1.7 core/timer.py
- `effective_use_cuda_graph(requested)`: `return bool(requested) and not torch.version.hip` (`timer.py:63`); `timing_mode()` (`:66-74`) derives `requested_use_cuda_graph`, `effective_use_cuda_graph`, `timing_execution_mode`, `timing_note` from that flag alone.
- Eviction: `_flush_l2_buffer_mb()` = `max(64, 2 * llc_bytes // MiB)` with `llc_bytes = last_level_cache_bytes()` and a 128 MiB fallback (`:101-104`); `last_level_cache_bytes()` (`tilebench/hardware.py:95-114`) returns `_LLC_BYTES[arch]`, raises `UncalibratedCacheError` for `_LLC_CALIBRATION_REQUIRED`, else the runtime `L2_cache_size`. `_flush_l2_cache` fills the buffer and synchronises (`:107-117`).
- Scope: `with proton.scope(proton_scope_name): runner()` (`:255-256`) wraps the whole `run()`; `_collect_gpu_kernel_ns` sums every CUDA/HIP descendant (`:177-188`).
- `_prepare_runner` (`:120-143`) runs `f` three extra times before capture (`:130-131`, inside the Proton session started at `:243`, outside the scope) and on any exception silently returns the eager lambda (`:142-143`). `report_benchmark` returns `{"mean": mean_ns / 1e6}` only (`:284`); nothing records whether capture succeeded. Verdict: `effective_use_cuda_graph` reflects the torch build, not actual capture success.
- `_find_scope_mean_ns` returns `total_ns / repeat` (`:191-199`): only the mean exists; no per-repeat samples (Proton `data="tree"` aggregates the scope) (inference for the Proton side).

### 1.8 Output namespace
- `LLM_GENERATED_ROOT = BENCHMARK_ROOT / "llm_generated"` (`tilebench/paths.py:29`); `_iter_dir` = `<root>/<op>/<model>/<effort>/iter_<N>` (`generate.py:48-52`), `final/` (`:55-56`), `run_summary.json` beside them (`:452`).
- `iter_N/` holds `prompt.md` (`:223`), `response.md` (`:237`), `impl_triton.py`/`impl_cutile.py` (`:267`), `llm_usage.json` (`:247`), copied `impl_torch.py` + trimmed `config.yaml` (`:154-156`), `feedback.json` (`:284`, `:349`, `:354`). `final/` is `rmtree`d and rebuilt every run (`:410-414`, `:428`). `run_summary.json` = `op, model, effort, max_iters, backends, history, best_clean, best_any, promotion` (`:438-451`).
- A rerun with the same (op, model, effort) reuses cached responses (`:227-230`) but rewrites `feedback.json`, `final/` and `run_summary.json`. The directory is git-ignored (`.gitignore:63`), absent in this worktree, and is the restore target of the archived campaign (`artifacts/manifest.json:2-9`; guarded by `tests/test_artifacts.py:263-271`). Verdict: v2 must write under a new root, never into `llm_generated/<op>/<model>/<effort>/`.

### 1.9 llm_client.py
- Provider by model prefix: `gpt*` → OpenAI, `claude*` → Anthropic, else `ValueError` (`tilebench/llm/llm_client.py:47-54`).
- OpenAI = Responses API, streaming: `self._client.responses.stream(model=..., input=input_msgs, reasoning={"effort": self.effort})` (`:124-128`), system as a `role: system` input (`:117-118`), TTFT on `response.output_text.delta` (`:131-134`), `get_final_response()` (`:141`).
- Anthropic = Messages API, streaming: `messages.stream(model, max_tokens=128000, messages=[user], thinking={"type": "adaptive"}, output_config={"effort": self.effort})` (`:148-154`, `:161`), `text_stream` (`:162`), `get_final_message()` (`:171`). Single-turn: the whole prompt is rebuilt each iteration (`:151`).
- Usage (`_extract_usage`, `:179-212`): OpenAI `usage.input_tokens`, `usage.output_tokens`, `output_tokens_details.reasoning_tokens`, `input_tokens_details.cached_tokens` (`:198-205`); Anthropic `input_tokens`, `output_tokens`, `cache_read_input_tokens`, no reasoning count (`:207-211`). No `cache_control` is sent and `cache_creation_input_tokens` is not read.
- Retries: SDK `max_retries=0`, `timeout=1800.0` (`:62`, `:69`); own loop `retries=3`, backoff `30 * 2**attempt` s (`:72-101`).
- No model ID is hard-coded in the client; defaults live in `generate.py:293` (`"gpt-5.5"`, help "gpt-5.5 / claude-opus-4-7" `:294`) and `generate_descriptions.py:124`; effort default `"xhigh"` (`llm_client.py:44`, `generate.py:297`).

### 1.10 NKI timing stack
- Two input modes: `PROFILE_INPUT_MODE_INSPECT = "runtime_inspect_real_inputs"`, `PROFILE_INPUT_MODE_CAPTURE = "tool_default"` (`tilebench/core/nki_timer.py:49-50`).
- Real-input path: engine (`engine.py:219-231`) → `profile_case_on_neuron` (`tilebench/core/nki_orchestrator.py:195-217`) saves the case's CPU inputs + reference as a bundle (`:254-263`), optional selector phase pins the autotune winner (`:269-293`), builds the spec with `warmup=int(warmup), repeat=int(repeat)` (`:311-312`), launches the profile worker (`:390-395`). The worker moves the bundle to the XLA device (`tilebench/core/nki_profile_worker.py:216`), runs once + verifies (`:169-179`), then `_timed_windows(run_once, warmup, repeat)` (`:188-189`), which records `[begin, end)` XLA execution indices via torch-xla's `ExecuteTime` count (`:103-129`). The parent ingests the runtime trace and calls `time_windows(executions, worker_result[target]["windows"])` (`nki_orchestrator.py:463`); `time_windows` (`nki_timer.py:179-229`) returns `mean, min, max, stdev, repeat (= len(windows)), per_iteration_ms, executions_per_iteration, per_model, method="neuron_rt_inspect", profile_input_mode`; the orchestrator adds `warmup, session_dir, parquet_dir` (`:499-500`) and audit fields (`:568-569`, `:605-608`). This is the only path that yields `nki_ms` (`:609`).
- Tool-default path: `profile_neff` (`nki_timer.py:258-303`) runs `neuron-explorer capture --profile-nth-exec={nth}` with `nth = max(2, warmup + 1)` (`:270-276`), returns `"repeat": 1`, `profile_input_mode="tool_default"` (`:293-303`). Used only for the `NKI_NEFF_PATH` override (`tilebench/core/nki_artifact.py:43`, `:264-269`; `nki_orchestrator.py:502-519`), reported as `nki_ok=False`, "correctness NOT verified" (`:594-604`), so `nki_ms` stays NaN (`:609`).
- Candidate timing in `NkiAutotuner` (`warmup=5, iters=10`, `tilebench/core/nki_autotune.py:299-300`) uses `CompiledKernel.benchmark` on numpy copies (`:317-359`) or a host wall-clock median (`:361-371`); it selects configs only and is replayed without timing in the profile worker (`:406-409`; `nki_profile_worker.py:17-19`).
- Verdict: `runtime_inspect_real_inputs` measures the REAL task inputs; `tool_default` and the autotuner's candidate numbers do not.

### 1.11 data/tensors.py
- `DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` (`tensors.py:11`); every generator takes `device=DEFAULT_DEVICE` (e.g. `:58`, `:69`).
- `expand_cases(operator_name, config)` (`:610-627`): priority `test_cases` > `case_grid` (+ `case_defaults`) > `case_preset`; `_expand_case_grid` is `itertools.product` over `_normalize_sequence` lists (`:26-39`), and `expr` values go through bare `eval(values["expr"])` (`:17`). `CASE_PRESETS` (`:54-56`); `get_generator` raises `ValueError` for unknown ops (`:788-791`); `infer_problem_size` (`:630-786`) uses `n`, then `shape`, then per-operator rules, then the product of all int params (`:780-786`).
- RNG: no `manual_seed`/`seed`/`torch.Generator` anywhere in `tensors.py`, `engine.py`, `tilebench/llm/*.py` or `scripts/run_bench.py` (grep empty). Inputs come from the unseeded global RNG: fresh values on every call, not reproducible across processes.

### 1.12 verifier.py
- `config_tolerance(verify_cfg: dict, arch: str | None) -> tuple[float | None, float | None]` (`verifier.py:59-66`): `cfg = {**verify_cfg, **(verify_cfg.get("arch_overrides") or {}).get(arch, {})}`; a key given nowhere is `None` = per-dtype default.
- `verify(output, reference, atol=None, rtol=None)` (`:44-56`): tuple/list outputs compared element-wise via `zip` (`:50-55`; a length mismatch is not detected, inference). `_verify_single` (`:23-41`) looks tolerances up by `output.dtype` (`:29`); integers are exact `(0, 0)` (`:10-13`); fp8 is cast to fp32 (`:34-36`); unknown dtypes get `1e-2/1e-2` (`:17-18`).

### 1.13 engine.py
- Cases: `expand_cases` (`engine.py:147`), optional `case_indices` (`:148-155`). Per case, `params` drop `dtype`/`block_size` (`:184`), `generate_inputs(**params, dtype=dtype)` (`:192`).
- Skip classification: generator or torch reference raising `(RuntimeError, TypeError, ValueError)` prints `Skipped: dtype=... not supported ...` and `continue`s (`:193-195`, `:200-203`); any other exception propagates (`tests/test_engine_case_skip.py:74`).
- Backend failure: `except Exception` → `*_ok=False`, `*_err=str(e)`, `*_ms=nan`, `*_stats=None` (`:270-275`, `:302-307`, `:334-339`); a verify failure keeps `*_err` and skips timing (`:263-269`). Record per case: `params, problem_size, dtype, torch_ms, torch_stats, <b>_ms/_stats/_ok/_err/_autotune_cfg, speedup_<b> = torch_ms/ms if ms > 0 else 0.0, timing` (`:372-403`).

### 1.14 Reusable public API
- `tilebench/hardware.py`: `device_info() -> DeviceInfo | None` (`:58`), `detect_arch() -> str | None` (`:74`), `supports_tmem()` (`:85`), `supports_tma()` (`:90`), `last_level_cache_bytes() -> int | None` (`:95`), `UncalibratedCacheError` (`:46`).
- `tilebench/provenance.py`: `source_state(repo=PACKAGE_ROOT) -> dict` (`:51`), `software_state() -> dict` (`:98`), `device_state(requested_label) -> dict` (`:115`), `collect(requested_label) -> dict` (`:127`), `SCHEMA` (`:38`).
- `tilebench/paths.py`: constants `PACKAGE_ROOT..PROBLEMS_ROOT` (`:23-35`), `REPO_ROOT` (`:37`), `OUTPUT_ROOT` (`:40`), `RESULTS_ROOT` (`:60`); `hardware_label(label)` (`:65`), `results_root/csv_dir/logs_dir/figures_dir/aggregate_dir/runs_dir(hardware)` (`:78-100`), `timing_log_path/autotune_log_path/provenance_log_path(hardware, operator, mode, backends)` (`:109-127`), `operator_dir(op)` (`:183`), `operator_config(op)` (`:188`), `list_operators()` (`:192`).
- `tilebench/backends.py`: `BACKEND_ORDER` (`:13`), `GPU_BACKENDS` (`:17`), `MODES` (`:20`), `canonical_backends(backends)` (`:23`), `parse_backends(text)` (`:32`), `backend_tag(backends)` (`:43`).
- `tilebench/core/timer.py`: `report_benchmark(f, args, kwargs, *, warmup, repeat, use_cuda_graph, proton_*, flush_l2, ...) -> {"mean": ms}` (`:208-223`), `timing_mode(requested)` (`:66`), `effective_use_cuda_graph` (`:53`), `_flush_l2_cache` (`:107`).

## 2. Reuse / replace / isolate

| Legacy item | Decision | Why |
|---|---|---|
| `core/timer.py` `report_benchmark`, `timing_mode`, `_flush_l2_cache` | REUSE as-is | Formal protocol (1/3, graph policy, LLC eviction) with tests; v2 passes its own `warmup/repeat` if the protocol differs. |
| `core/verifier.py` `verify`, `config_tolerance` | REUSE as-is | v2 must call `config_tolerance(cfg["verify"], detect_arch())` like the engine (`engine.py:133`), fixing 1.6. |
| `core/engine.py` | REUSE with wrapper | Backend dispatch/skip rules are right, but it imports fixed `tilebench.benchmarks.operators.<op>` modules (`:79-109`); v2 loads generated modules from its own dir. |
| `core/metrics.py` `compute_derived`, `load_peak_config` | REUSE as-is | Same expression context as `roofline._eval_ctx`; `load_peak_config(gpu)` is label-driven (`:12-21`), unlike `roofline.load_peak("B200")`. |
| `core/dtypes.py` | REUSE as-is | Shared dtype names/sizes. |
| `core/cutile_autotune.py`, `core/tilelang_log.py` | LEGACY-ONLY | Autotune is forbidden to the LLM (`evaluator_runner.py:152-160`); unrelated to v2 generation. |
| `core/nki_*.py` | REUSE with wrapper (if NKI is in v2) | Real-input timing only through `profile_case_on_neuron`; needs `logs_dir` paths (`engine.py:44-52`). |
| `hardware.py`, `provenance.py`, `paths.py`, `backends.py` | REUSE as-is | See 1.14; add v2 path helpers to `paths.py` rather than string concatenation. |
| `data/tensors.py` `expand_cases`, `get_generator`, `infer_problem_size` | REUSE as-is | Case selection must come from `expand_cases()` (1.4); add seeding in v2 around the call (1.11). |
| `llm/generate.py` | REPLACE in v2 | Namespace, trimming, promotion and feedback fields are protocol-specific (1.1, 1.4, 1.8). |
| `llm/prompt_builder.py` `parse_response`, `_read_skill` | REUSE with wrapper | Parsing/frontmatter logic is fine; v2 must fail loudly on a missing skill and own its prompt layout. |
| `llm/prompt_builder.py` prompts, `_format_feedback`, `_format_trajectory` | REPLACE in v2 | They leak score fields (1.5) and embed legacy wording. |
| `llm/llm_client.py` `LLMClient`, `_extract_usage` | REUSE with wrapper | Provider calls and usage extraction are sound; v2 fixes effort default and records model IDs explicitly. |
| `llm/evaluator.py` subprocess isolation | REUSE with wrapper | Env-var contract (`:41-47`) and timeouts are reusable; point it at a v2 runner. |
| `llm/evaluator_runner.py` | REPLACE in v2 | 5/20 protocol, `B200` peak, default tolerances, 52 generator calls, no graph (1.3, 1.6). Keep `_FORBIDDEN_PATTERNS`/`_scan_for_forbidden` (`:129-171`) by import. |
| `llm/roofline.py` `roofline_pct` | REUSE with wrapper | Pure function; pass `load_peak(gpu_label)` from the run's `--gpu`. |
| `llm/system_prompt.md`, `framework_guide.md` | LEGACY-ONLY | Hard-code 10 iters, 5+20, B200 (section 3); paper-campaign prompt inputs must stay byte-identical. |
| `llm/generate_descriptions.py`, `tilebench/problems/*_current.md` | LEGACY-ONLY (read-only input) | Descriptions are prompt inputs; never rewritten by v2. |
| `skills/{triton,cutile}-guide/SKILL.md` | LEGACY-ONLY (read-only input) | Frozen API references of the paper campaign. |

## 3. Hard-coded protocol values

| Value | Location |
|---|---|
| `BACKENDS = ("triton", "cutile")` | `generate.py:45` |
| `--model` default `"gpt-5.5"`; help `gpt-5.5 / claude-opus-4-7` | `generate.py:293-294`; `generate_descriptions.py:124` |
| `--effort` default `"xhigh"`, choices `minimal/low/medium/high/xhigh` | `generate.py:297-298`; `llm_client.py:44`; `generate_descriptions.py:129` |
| `--max-iters` default 10 | `generate.py:301`; "10-iteration" in `system_prompt.md:8`, `framework_guide.md:35`, `:200`, `:327` |
| `impl_triton.py` / `impl_cutile.py` file names | `generate.py:160-161`, `:206`; `prompt_builder.py:38`, `:552-554`; `framework_guide.md:10-11`, `:338` |
| warmup 5 / repeat 20 | `evaluator_runner.py:57-58`, `:269`, `:275`; `framework_guide.md:319-320` |
| `load_peak("B200")` | `evaluator_runner.py:351`; `roofline.py:20`; `B200`/`sm_100` in `system_prompt.md:3`, `framework_guide.md:166`, `:180`, `:218`, `:229` |
| `roofline_pct` capped at 1.0 for scores | `evaluator_runner.py:431` |
| per-case cap 900 s; overall 7200 s | `evaluator.py:23-24`; `evaluator_runner.py:324` |
| `_FORBIDDEN_PATTERNS` (torch ops + autotune APIs) | `evaluator_runner.py:129-161` |
| Anthropic `max_tokens=128000`; SDK timeout 1800 s; retries 3, backoff 30 s | `llm_client.py:150`, `:62`, `:69`, `:72-73` |
| Triton version "3.6.0" in the prompt | `prompt_builder.py:91` |
| Feedback truncation: 10 verify failures, 20 combos, 3000-char timeout text, 1000-char error | `prompt_builder.py:489`, `:508`, `:478`; `evaluator_runner.py:395` |
| stdout/stderr tails 2000/4000 chars | `evaluator.py:67-68`, `:75`, `:84-85` |
| Output root `benchmarks/llm_generated` | `paths.py:29`; `generate.py:48-56`, `:452` |
| Timer fallback 128 MiB LLC, 64 MB floor | `timer.py:103-104` |

## 4. Existing tests touching the LLM track or timer semantics

- `tests/test_measurement_protocol.py`: every operator config and the framework fallback are warmup=1/repeat=3 (`:44-54`); every Triton/TileLang autotune site passes `warmup=1, rep=3` (`:71-88`); candidate lists match a snapshot (`:147-163`).
- `tests/test_timing_policy.py`: `effective_use_cuda_graph`/`timing_mode` on NVIDIA vs ROCm (`:30-56`); fake Proton fixture `no_gpu_timer` (`:62-90`); ROCm never captures a graph, eager runner, NVIDIA replays (`:99-131`); eviction 253 MB on B200 and 512 MB on CDNA3 (`:135-146`); engine announces fallback once and records `timing` (`:179-267`).
- `tests/test_hardware.py`: arch detection and flush sizes (`:50-186`), including CDNA3 refusing to flush uncalibrated.
- `tests/test_verify_tolerance.py`: `config_tolerance` semantics and that the engine verifies with the detected arch (`:22-61`).
- `tests/test_engine_case_skip.py`: ValueError-skip vs propagation (`:51-77`).
- `tests/test_results_layout.py`: `tilebench/problems/` has one `_current.md` per operator and `prompt_builder._problem_desc_path` points there (`:66-74`); frozen CSV schema (`:127-139`).
- `tests/test_profiling_metadata.py`: an installed package ships `tilebench/llm/prompt_builder.py`, `framework_guide.md` and 45 problem files (`:247-261`).
- `tests/test_artifacts.py`: `llm_generated/` is git-ignored and untracked while `tilebench/llm/generate.py` is tracked (`:263-271`); fetch/package of the `llm-aacl2026` archive (`:69-254`).
- `tests/test_nki_timer_windows.py`, `test_nki_profile_orchestration.py`, `test_nki_autotune_replay.py`, `test_nki_artifact_identity.py`, `test_nki_profile_spec.py`: window timing, exact-winner replay, artifact identity, spec IDs (all CPU-only).
- `tests/test_provenance.py`: `source_state`/`software_state`/`device_state`/`collect` and the run_bench sidecar (`:44-258`).
- No test exercises `generate.py`, `evaluator*.py`, `llm_client.py`, `roofline.py`, or the prompt text itself.
