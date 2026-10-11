# Trn2 NKI LLM Base (protocol revision 4) — preparation, launch and campaign record

Branch `exp/llm-trn2` (worktree `tb_wt/_llm_trn2`). Revision-4 protocol unchanged:

- 5 rounds, exactly 1 generation per round, no same-round repair;
- every candidate is evaluated on all 20 frozen cases of the operator's representative dtype, 1 warmup + 3 timed;
- generators GPT `gpt-6.1-sol` (reasoning xhigh) and Claude `claude-opus-5-5` (adaptive thinking, effort xhigh);
- `review_required` items are adjudicated by Claude Code under the existing checker rules.

Nothing here changes B200 / GH200 / MI300X data or their branches. The formal campaign finished 2026-10-11 05:15Z
(360/360 rounds); see "Campaign outcome" at the end.

## Environment

trn2.3xlarge, 1 Trainium2 chip, LNC=2: 4 logical / 8 physical NeuronCore-v3. Python environment
`/home/ubuntu/workspace/native_venv`:

- NKI Beta 5 stack: nki 0.7.0b1, neuronx-cc 2.0.404830.0a0, torch 2.14.0+cpu, torch-neuronx 2.14.3;
- openai 2.37.0 / anthropic 0.103.1 (the B200 versions) were added. The install was purely additive: 14 new packages,
  nothing upgraded or removed.

## Components

| piece | file |
|---|---|
| NKI Reference Skill (public) | `skills/reference/nki/0.7.0b1/` (SKILL.md, COMPATIBILITY.md, SOURCES.md, licences, evidence/) |
| Trn2 Device Context | `skills/device/Trn2/2026-10-10/` (SKILL.md, manifest.json, runtime_capture.json) |
| eligibility (frozen) | `tilebench/llm/v2/manifests/trn2_eligibility.json` (built by `scripts/llm_v2/build_trn2_eligibility.py`; read by `tasks/support.py`) |
| torch baseline (reporting speedup) | `artifacts/llm_v2/baselines/Trn2/torch_baseline_Trn2_casesets_2026-10-10.json` |
| scoring table (frozen profile, 720 ok) | `artifacts/llm_v2/scoring/Trn2/scoring_table_Trn2_rev4_casesets_2026-10-10.json` |
| native evaluator | `tilebench/llm/v2/evaluation/neuron.py`, `neuron_worker.py`; timing adapter `neuron_native_trace` |
| freeze (frozen, T1/T2/T3 recorded) | `tilebench/llm/v2/manifests/frozen/Trn2_base_rev4_2026-10-10.json` (`scripts/llm_v2/build_trn2_freeze.py`) |
| validation evidence | `artifacts/llm_v2/preflight/Trn2/` (final readiness: `FINAL_READINESS_2026-10-10.json`) |

## Eligibility: 36 eligible / 9 excluded (720 cases)

A task is eligible only when all 20 frozen cases of its representative dtype (1) have a valid formal PyTorch eager
baseline in `results/TRN2/csv` (from main #322) and (2) have a supported empirical SOL target (status `ok` under the
declared mode and the frozen Trn2 profile; launch decision T2). The dtype is never changed. Each exclusion carries an
`exclusion_class` (manifest schema `tilebench-llm-v2-trn2-eligibility/2`):

- `cpu_fallback`: argmax, 2d_max_pooling, radix_sort;
- `baseline_correctness_failure`: flash_attention;
- `no_eager_baseline`: block_sparse_attention;
- `baseline_case_failure`: gaussian_blur (1/20 failed), reverse_array (2/20 failed), top_k_selection (1/20 failed);
- `scoring_peak_unavailable`: matmul_int8. It runs on the device (20/20 valid baselines, not a CPU fallback), but the
  declared mode `mma_int8_i32acc` has no calibrated Trn2 ceiling (no int8 TensorE input type). No bf16 or theoretical
  peak is substituted.

Kept, each with 20/20 valid baselines and 20/20 `ok` targets: interleave fp16, matrix_transpose fp16 and
matmul_fp32_fp16_fp8 fp16.

## Native evaluator (Trn2)

- **Worker.** `evaluation.worker.run_case` and the suite rules are reused unchanged. Only the device pieces are new:
  - `torch.device("neuron")` via `tilebench.core.neuron_native`, with no torch-xla and torch.compile disabled;
  - inputs are generated on the CPU with the case seed and moved to the device;
  - the reference is `impl_torch.run` on CPU copies (the formal Trn2 definition);
  - the three numerical checks (fresh / refill / repeat) have the same mutation and aliasing rules, compared on the CPU.
- **Timing.**
  - Per call: the device busy sum of every Neuron execution of one `run()`, read from the runtime trace
    (NeuronConfig RUNTIME, `nc_exec_running`). The per-physical-core copies of one execution are merged by
    (model_id, flow_id), so LNC=2 is never double counted.
  - The first execution (compile) is excluded; then 1 warmup and 3 timed calls.
  - A call without a device execution, or with different executables between calls, is a `timing_error`. Host
    wall-clock is never used.
- **Host fallback.** `torch_neuronx` op tracking runs around every candidate call. A CPU fallback op or a
  device-to-host copy is a `contract_violation`, which enforces system rule 4.
- **Hidden search.** Instances of `tilebench.core.nki_autotune` are a `contract_violation`.
- **Isolation.**
  - bwrap is used when it works. On this host Ubuntu 24.04 AppArmor blocks unprivileged user namespaces
    (`kernel.apparmor_restrict_unprivileged_userns=1`; bwrap is not installed; no sudo).
  - The fallback is a docker container with the bwrap allowlist and the same isolation probe:
    - system dirs, the venv prefix and its editable roots, and the repository are mounted read-only, with the same
      data masks;
    - only the task's `impl_torch.py` is visible;
    - `/home`, `/root`, `/tmp` and `/var/tmp` are private tmpfs;
    - no network, own PID namespace, all capabilities dropped, no-new-privileges, the caller's uid;
    - `/dev/neuron*` is passed through, the environment is scrubbed, and compile caches are per candidate;
    - memory limit 100 GB.
  - The Trn2 device lock serializes evaluations.
- **Fingerprint.** The Trn2 fingerprint adds the neuron evaluator and core sources and the torch / torch-neuronx /
  nki / neuronx-cc versions. GPU fingerprints are byte-identical to before.

## Shared-manifest delta (Trn2 only)

- `study.yaml`:
  - `support_matrix.Trn2.timing_adapter` → `neuron_native_trace`;
  - `dsls.nki.reference_version` → `0.7.0b1`;
  - `device_snapshots.Trn2` → `2026-10-10`;
  - `torch_baselines.Trn2` added.
- `calibration.yaml`: Trn2 profile (frozen by T1; only the Trn2 entry differs).
- `skills/manifest.json`: the two new assets.

The Trn2 study config hash therefore differs from the B200 one. **Do not merge these changes into exp/llm-b200 /
exp/llm-gh200 / exp/llm-mi300x while their campaigns run** (it would change their config hash and refuse resume).
`tests/llm_v2/test_frozen_rev4.py` checks the B200 freeze on the B200 view of the study.

## SOL calibration and scoring readiness

- **Ceiling basis.** The Trn2 campaign uses the empirical calibration `Trn2-20261010T012011Z-94406dcb`, measured
  earlier and not re-run. Its file sha256 is `a2b810f7...`, and it is registered in `calibration.yaml` as `frozen`
  (launch decision T1). The architecture-theoretical audit (`artifacts/paper_figures/sol_theoretical_audit/hardware/TRN2`)
  is for the paper only and is not used for scoring.
- **Scoring table.** 720 rows (36 tasks x 20 cases), all `ok`. Every included case also has a valid stored torch
  baseline (independent cross-check in `FINAL_READINESS_2026-10-10.json`: 720/720, no problem).
- **Measured modes in use:**
  - `fp32_fma_vector` 0.747 and `fp16x2_fma_vector` 0.860 TFLOP/s (VectorE + ScalarE + GpSimd probe);
  - `mma_fp16_f32acc` 146.8 and `mma_tf32_f32acc` 121.4 TFLOP/s;
  - `hbm_stream_bw` 672.3 GB/s;
  - `memory_only` / `no_compute_term` as declared.
- **Known scoring caveats, reviewed before launch** (shared definitions, not changed here; neither invalidates the
  formal scores):
  - **RMSNorm.** The frozen Q = 3·n·dtype_size (two sweeps of x plus one write) is 1.4996x the compulsory
    (2n + K)·dtype_size on all 20 cases. Both are memory-bound. The torch baseline's efficiency is 0.24–0.64 under the
    frozen Q (0.16–0.43 under the compulsory Q). The formal manual NKI results (`results/TRN2/csv`, `nki_ms`)
    reach 0.29–0.80 under the frozen Q, none above 1.
    - It is a uniform per-operator scale of the definition B200/GH200/MI300X also use. Within-operator comparisons
      are unchanged, and efficiency > 1 is kept as an audit flag.
    - T_emp is never used during a run (feedback carries only valid_cases and latency_ms_geomean).
    - A correction, if wanted, is a central `arithmetic_modes` decision (Q override, as M4 did for radix_sort) for every
      device at once, after the running campaigns finish. `arithmetic_modes.yaml` is pinned by their scoring bindings.
  - **linear_self_attention (fp32, `mma_tf32_f32acc`).** The contract allows TF32 operands for S and the numerator.
    All 20 Trn2 cases are memory-bound (T_emp = Q / BW_emp), so the TF32 rate never sets the target. Even the IEEE fp32
    control rate would move only the 5 D=256 cases, by at most x1.16.
  - The fp32 FMA vector peak comes from a three-engine probe.
- **Reporting note (post hoc, not a launch blocker).** `devtools.recompute_metrics` passes all 45 representative
  operators to `metrics.efficiency.aggregate` as `eligible`. For Trn2 the 9 predeclared exclusions would count as
  `missing`: the exact E̅(B) would be None and the lower bound scaled by 36/45.
  - Minimum correction: build the group's eligibility from `support.task_table` for its device (eligible | unsupported).
  - This is a no-op for B200/GH200/MI300X, which have 45/45 eligible tasks.
  - Needed before the Trn2 E̅(B) report; not changed without approval.

## Validation results (2026-10-10; evidence `artifacts/llm_v2/preflight/Trn2/VALIDATION_2026-10-10.json`)

- **Real-chain end-to-end run** (`scripts/llm_v2/trn2_e2e_validation.py`, run type validation, scripted provider, no
  API request), vector_add fp16, all 20 cases, docker sandbox:
  - round 1 `valid` on 20/20 cases, geomean 0.0996 ms;
  - worker wall time 104.6 s; one device execution per call; no host fallback;
  - a `--resume` sent 0 requests and changed none of the 18 archived files.
- **Negative candidates:**
  - host round-trip and on-device `torch.sort` (CPU fallback) → `contract_violation`;
  - wrong result → `numerical_error`;
  - input aliasing → `contract_violation`;
  - bad tile → `compile_error` (suite stopped).
- **Multi-execution `run()`:** the busy sum adds the executions and excludes the idle gap between them.
- **Isolation probe (docker): ok.** The clean NKI candidate is `audit_only` under the unchanged checker.
- **Prompts:** all 37 pre-T2 tasks (the 36 included ones among them) render with `reference:nki@0.7.0b1` + `device:Trn2@2026-10-10`. None contains a manual NKI
  line, a baseline latency, a calibration/SOL value or a speedup/efficiency term.
- **Tests:** `tests/llm_v2`: 268 passed. The 2 failures are the same environment failures as clean HEAD (no `triton`
  on this host).
- **CPU reference cost:** bitonic_sort needs about 815 s per 20-case suite, inside the frozen 3600 s worker limit;
  every other operator needs ≤ 35 s.

## Owner decisions (2026-10-10, recorded in the freeze)

- **T1.** Freeze the existing empirical profile `Trn2-20261010T012011Z-94406dcb`; no recalibration; the theoretical audit
  is separate from revision-4 scoring.
- **T2.** Exclude matmul_int8 for `scoring_peak_unavailable` (not CPU fallback): 36 eligible / 9 excluded, 20 cases per
  operator, 2 models x 5 rounds = 360 generation rounds; representative dtypes and frozen case sets unchanged.
- **T3.** Docker is the formal Trn2 sandbox (the validated docker isolation path; isolation is never disabled; GPU
  evaluators unchanged).

Final formal preflight (`--live --run-type formal --conditions base`): `ok`, no blocker, for `--provider openai` and
`--provider anthropic`. The one warning is the docker backend (T3).

## Campaign size and launch plan

| | trajectories | generation rounds |
|---|---|---|
| 1. acceptance (`--operators vector_add --dtypes fp16`) | 2 (gpt, claude) | 10 |
| 2. full set (`--resume`, no filter) | 72 = the 2 above (resumed, no new request) + 70 new | 360 in total (10 + 350) |

Trajectory directories are deterministic per (campaign, task, model, condition):

- an existing trajectory is only resumed: archived responses are reused, never re-requested;
- without `--resume` an existing trajectory is refused, never restarted;
- the 9 excluded operators are skipped as predeclared `unsupported`.

## API authentication

- GPT-6.1 Sol reads `OPENAI_API_KEY`, and Claude Opus 5.5 reads `CLAUDE_API_KEY` (`models.yaml` `api_key_env`; read
  from the environment when the provider is built).
- `systemd-run --setenv=NAME` (no value) copies the variable from the invoking shell, so a key is never on a command
  line, in a file of the repository, or in a log.
- Never run `systemctl --user show -p Environment` on the unit: it would print the keys.

## Launch (run from `tb_wt/_llm_trn2`)

Launched 2026-10-10 16:36Z on the owner's go. The owner skipped the separate vector_add acceptance run, so the full set
was launched directly. The command is also the restart / recovery command:

```bash
cd /home/ubuntu/tb_wt/_llm_trn2
PY=/home/ubuntu/workspace/native_venv/bin/python
systemd-run --user --unit=tb-trn2-nki-base-full -p MemoryMax=16G -p OOMPolicy=continue \
  --working-directory=/home/ubuntu/tb_wt/_llm_trn2 --setenv=PATH --setenv=OPENAI_API_KEY --setenv=CLAUDE_API_KEY \
  $PY -m tilebench.llm.v2 schedule --device Trn2 --dsls nki --models gpt claude --condition base \
    --campaign formal_trn2_nki_base_rev4_2026-10-10 --run-type formal --isolation docker --resume \
    --executor "claude-code (Trn2 NKI Base)"
# monitor / pause
journalctl --user -u tb-trn2-nki-base-full -f
$PY -m tilebench.llm.v2 coverage outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10
$PY -m tilebench.llm.v2 review-queue outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10
touch outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10/STOP      # graceful pause (in-flight requests finish)
```

Two flags are required. Both were learned on launch day:

- **`--setenv=PATH`.** The systemd user manager's PATH lacks `/opt/aws/neuron/bin`. Without `neuron-ls`, host-arch
  detection gives None and preflight refuses both tracks. The first launch failed this way, before any trajectory or
  request.
- **`--resume`, always, even on a fresh campaign.** After a `review-resolve`, the scheduler continues the trajectory in
  process by resuming it. Without `--resume` that continuation raises "already holds a trajectory". This happened once
  (gpt batched_matmul, round 1). The fix was a graceful STOP drain (no request lost) and a `--resume` relaunch at
  17:41Z: all 72 trajectories resumed, 0 new, 0 re-requested.

The unit cap covers the scheduler only. The evaluation container runs under dockerd with its own 100 GB limit
(`evaluation/neuron.py`), so 16 + 100 GB stays under the host's 124 GiB.

API generation runs concurrently (adaptive per provider, global evaluation backlog 16). Device evaluation is serial
under the Trn2 lock (one candidate on one logical NeuronCore at a time). Tokens and USD are recorded per attempt under
the frozen pricing snapshot.

## Campaign outcome (complete 2026-10-11 05:15Z)

The publication export (`python -m tilebench.llm.v2 export-publication`) is at
`artifacts/llm_v2/formal_trn2_nki_base_rev4_2026-10-10/`. It holds every round of all 72 trajectories, 0 withheld,
because every injected component is publishable. Further files beside it:

- `campaign_report.{json,md}`, `cost_report.json` and `coverage.json`, written by the CLI commands of the same names.
- `scheduler/`: the three `scheduler_summary_*.json` files and the `telemetry/` jsonl (concurrency and scheduler events),
  with `SHA256SUMS`. The export globs only `summary_*.json`, so these were copied by hand.

The run cache under `outputs/` stays git-ignored.

Execution:

- 72/72 trajectories `complete`, 360/360 rounds, exactly one attempt per round.
- No same-round regeneration and no duplicate request.
- Unit exit 0; scheduler `errors: []`, `refused_tracks: []`.
- The only interruption was the `--resume` relaunch described above.

| | rounds | valid | compile_error | format_error | runtime_error | numerical_error | contract_violation | tasks with ≥1 valid round | logical tokens | est. USD |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6.1-sol | 180 | 141 | 26 | 0 | 8 | 4 | 1 | 35/36 | 6,501,157 | 29.22 |
| claude-opus-5-5 | 180 | 132 | 17 | 21 | 2 | 6 | 2 | 33/36 | 19,886,926 | 273.49 |
| total | 360 | 273 | 43 | 21 | 10 | 10 | 3 | | 26,388,083 | 302.71 |

Compliance reviews: 19 `review_required` rounds were adjudicated by `claude-code` from the source, contract, rules
and evidence only. 18 were found compliant and 1 a violation. Each decision and its rationale is in the trajectory's
`reviews.jsonl`.

- Compliant:
  - in-kernel NKI access-pattern views (`.permute`) used by in-kernel DMA transposes (gpt batched_matmul r1–r5);
  - shape-only `_LAST_CONFIG` metadata (gpt flash_decode r1–r5, claude flash_decode r3–r5);
  - an import-time `inspect.signature` probe of the installed NKI API (claude jacobi_stencil_2d r3–r5);
  - scale indexing by floor division by the block side (gpt weight_dequant r2, r5).
- Violation: claude dropout r4. `run()` tries up to five kernel variants or configurations in a try/except cascade and
  caches the outcome in `_MODE_BY_N`. That is the runtime configuration search forbidden by system rules 3/4 and the
  dropout contract.

Reporting caveat, open, to be fixed after the campaign on the owner's go: `devtools.recompute_metrics` passes all 45
representative operators to `aggregate` as eligible. On Trn2 the exact E̅(B) then comes out None, and the lower bound
is scaled by 36/45. The minimum fix is a device-aware eligible / unsupported denominator taken from
`support.task_table`. It is a no-op on B200, GH200 and MI300X, where all 45 are eligible. Do not quote Trn2 `metrics`
output until that fix lands. `campaign_report`, `cost_report` and `coverage` do not use that aggregation.

Git provenance: the campaign records name launch commit `a713419b`. Before the branch was first pushed, the four
unpushed commits from `01398d75` on were rewritten. The only change was redacting an EC2 instance id from
`skills/reference/nki/0.7.0b1/evidence/smoke_results.json` and updating that file's hash in the skill's
`manifest.json`. The launch commit is published as `1c3cc97e`, with every other path byte-identical. The full map is in
`artifacts/llm_v2/formal_trn2_nki_base_rev4_2026-10-10/GIT_REWRITE_2026-10-11.json`. SKILL.md (the injected text) and
every pinned freeze, scoring and trajectory hash are unchanged.
