# Trn2 NKI LLM Base (protocol revision 4) — preparation record

Branch `exp/llm-trn2` (worktree `tb_wt/_llm_trn2`). Revision-4 protocol unchanged:

- 5 rounds, exactly 1 generation per round, no same-round repair;
- every candidate is evaluated on all 20 frozen cases of the operator's representative dtype, 1 warmup + 3 timed;
- generators GPT `gpt-6.1-sol` (reasoning xhigh) and Claude `claude-opus-5-5` (adaptive thinking, effort xhigh);
- `review_required` items are adjudicated by Claude Code under the existing checker rules.

Nothing here changes B200 / GH200 / MI300X data or their branches.

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
| scoring table (candidate profile) | `artifacts/llm_v2/scoring/Trn2/scoring_table_Trn2_rev4_casesets_2026-10-10.json` |
| native evaluator | `tilebench/llm/v2/evaluation/neuron.py`, `neuron_worker.py`; timing adapter `neuron_native_trace` |
| freeze (prepared) | `tilebench/llm/v2/manifests/frozen/Trn2_base_rev4_2026-10-10.json` (`scripts/llm_v2/build_trn2_freeze.py`) |
| validation evidence | `artifacts/llm_v2/preflight/Trn2/` |

## Eligibility: 37 eligible / 8 excluded

A task is eligible only when all 20 frozen cases of its representative dtype have a valid formal PyTorch eager baseline
in `results/TRN2/csv` (from main #322). The dtype is never changed.

Excluded operators:

- argmax, 2d_max_pooling, radix_sort: CPU fallback;
- flash_attention: baseline correctness failure;
- block_sparse_attention: no pure eager baseline;
- gaussian_blur (1/20 failed), reverse_array (2/20 failed), top_k_selection (1/20 failed).

Kept, each with 20/20 valid baselines: interleave fp16, matrix_transpose fp16, matmul_fp32_fp16_fp8 fp16 and
matmul_int8 int8.

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
- `calibration.yaml`: Trn2 candidate profile.
- `skills/manifest.json`: the two new assets.

The Trn2 study config hash therefore differs from the B200 one. **Do not merge these changes into exp/llm-b200 /
exp/llm-gh200 / exp/llm-mi300x while their campaigns run** (it would change their config hash and refuse resume).
`tests/llm_v2/test_frozen_rev4.py` checks the B200 freeze on the B200 view of the study.

## SOL calibration and scoring readiness

- **Ceiling basis.** The Trn2 campaign uses the empirical calibration `Trn2-20261010T012011Z-94406dcb`, measured
  earlier and not re-run. Its file sha256 is `a2b810f7...`, and it is registered in `calibration.yaml` as
  `candidate`. The architecture-theoretical audit (`artifacts/paper_figures/sol_theoretical_audit/hardware/TRN2`) is
  for the paper only and is not used for scoring.
- **Scoring table.** 740 rows (37 tasks x 20 cases): 720 `ok` and 20 `peak_unavailable`. The 20 are matmul_int8:
  mode `mma_int8_i32acc` is unsupported on NeuronCore-v3 (no int8 TensorE). No bf16 peak is substituted.
  matmul_int8 is execution-eligible but scoring-unavailable.
- **Measured modes in use:**
  - `fp32_fma_vector` 0.747 and `fp16x2_fma_vector` 0.860 TFLOP/s (VectorE + ScalarE + GpSimd probe);
  - `mma_fp16_f32acc` 146.8 and `mma_tf32_f32acc` 121.4 TFLOP/s;
  - `hbm_stream_bw` 672.3 GB/s;
  - `memory_only` / `no_compute_term` as declared.
- **Known scoring caveats** (central issues, not changed here):
  - the RMSNorm traffic formula (frozen Q = 3n overcounts, so efficiency can exceed 1);
  - linear_self_attention (fp32) is declared `mma_tf32_f32acc`, and its ceiling is the compiler auto-cast TF32 path;
  - the fp32 FMA vector peak comes from a three-engine probe.

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
- **Prompts:** all 37 render with `reference:nki@0.7.0b1` + `device:Trn2@2026-10-10`. None contains a manual NKI
  line, a baseline latency, a calibration/SOL value or a speedup/efficiency term.
- **Tests:** `tests/llm_v2`: 268 passed. The 2 failures are the same environment failures as clean HEAD (no `triton`
  on this host).
- **CPU reference cost:** bitonic_sort needs about 815 s per 20-case suite, inside the frozen 3600 s worker limit;
  every other operator needs ≤ 35 s.

## Open owner decisions (formal preflight: exactly T1 + T2 block)

- **T1. Freeze the Trn2 empirical profile.** Run
  `calibration-register --device Trn2 --profile artifacts/llm_v2/calibration/Trn2/Trn2-20261010T012011Z-94406dcb/profile.json --status frozen --by "<owner>"`.
- **T2. matmul_int8 (execution-eligible, no empirical int8 ceiling). Choose:**
  - (a) keep it as an execution-only task: speedup is reported, efficiency is unavailable. This needs a small scoring-gate
    change that accepts an explicitly declared `scoring_unavailable` Trn2 task. It is shared scoring policy, so it is
    not implemented without approval.
  - (b) exclude it from Trn2: 36 tasks, 360 generations.
- **T3. Accept the docker isolation backend** for formal Trn2 runs. Alternative: install bubblewrap and allow
  user namespaces (needs sudo).

## Launch (after T1 + T2; run from `tb_wt/_llm_trn2`)

```bash
cd /home/ubuntu/tb_wt/_llm_trn2
PY=/home/ubuntu/workspace/native_venv/bin/python
# 0. after the owner decisions: refresh the scoring table and the freeze record, then re-check
$PY -m tilebench.llm.v2 scoring-table --device Trn2 --dsl nki --out artifacts/llm_v2/scoring/Trn2/scoring_table_Trn2_rev4_casesets_2026-10-10.json
$PY scripts/llm_v2/build_trn2_freeze.py --decided T1="<owner, date>" T2="<owner, date, choice>" T3="<owner, date>"
$PY -m pytest tests/llm_v2 -q
$PY -m tilebench.llm.v2 preflight --devices Trn2 --live --run-type formal --conditions base --provider openai
$PY -m tilebench.llm.v2 preflight --devices Trn2 --live --run-type formal --conditions base --provider anthropic
# 1. launch acceptance: both models, vector_add only, formal identity (API keys pass through by name, never on the command line)
systemd-run --user --unit=tb-trn2-nki-base -p MemoryMax=110G -p OOMPolicy=continue \
  --working-directory=/home/ubuntu/tb_wt/_llm_trn2 --setenv=OPENAI_API_KEY --setenv=CLAUDE_API_KEY \
  $PY -m tilebench.llm.v2 schedule --device Trn2 --dsls nki --models gpt claude --condition base \
    --campaign formal_trn2_nki_base_rev4_2026-10-10 --run-type formal --isolation docker \
    --operators vector_add --dtypes fp16 --executor "claude-code (Trn2 NKI Base)"
# 2. full task set under the same identity
systemd-run --user --unit=tb-trn2-nki-base-full -p MemoryMax=110G -p OOMPolicy=continue \
  --working-directory=/home/ubuntu/tb_wt/_llm_trn2 --setenv=OPENAI_API_KEY --setenv=CLAUDE_API_KEY \
  $PY -m tilebench.llm.v2 schedule --device Trn2 --dsls nki --models gpt claude --condition base \
    --campaign formal_trn2_nki_base_rev4_2026-10-10 --run-type formal --isolation docker --resume \
    --executor "claude-code (Trn2 NKI Base)"
# monitor / pause
$PY -m tilebench.llm.v2 coverage outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10
$PY -m tilebench.llm.v2 review-queue outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10
touch outputs/llm_v2/formal_trn2_nki_base_rev4_2026-10-10/STOP      # graceful pause
```

API generation runs concurrently (adaptive per provider, global evaluation backlog 16). Device evaluation is serial
under the Trn2 lock (one candidate on one logical NeuronCore at a time). Tokens and USD are recorded per attempt under
the frozen pricing snapshot.
