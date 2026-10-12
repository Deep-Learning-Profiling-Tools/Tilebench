# LLM Base campaigns of the four devices (paper RQ3)

`scripts/paper_figures/llm_rq3.py --raw <extracted archives>` writes everything in this directory. It reads only
existing data, makes no API or GPU call, and changes neither the frozen framework nor any device branch.

## Inputs

| input | source |
|---|---|
| campaign trajectories (3,510 rounds) | HF `bcui2/TileArena-LLM-Raw` (private): `B200/formal_b200_base_rev4_reffix_2026-10-07`, `GH200/formal_gh200_base_rev4_reffix_2026-10-07`, `MI300X/formal_mi300x_triton_base_rev4_2026-10-09`, `Trainium2/formal_trn2_nki_base_rev4_2026-10-10` (`.tar.zst`; sha256 in `llm_rq3_manifest.json`) |
| frozen case sets, representative dtypes, price snapshot | `tilebench/llm/v2/manifests/{case_sets,representative_dtypes,api_pricing}.yaml`, byte-identical on the four branches |
| T_SOL | frozen empirical scoring table of each device (`artifacts/llm_v2/scoring/<device>/scoring_table_*_rev4_casesets_*.json`) at `exp/llm-b200` ac183b88, `exp/llm-gh200` 436d033f, `exp/llm-mi300x` be5f8111, `exp/llm-trn2` 7f2f0ca2; every trajectory's `scoring_binding` is checked against it |
| Trn2 eligibility (36 of 45) | `tilebench/llm/v2/manifests/trn2_eligibility.json` (exp/llm-trn2) |
| human kernels | `results/{B200,GH200,MI300X}/csv/<op>_autotune.csv` (this branch); `results/TRN2/csv/<op>_default.csv` and `autotune_selected_cases/nki_autotune_largest_case_per_dtype.csv` at origin/main 00a9751d |

## Metric (paper definition; differs from `tilebench.llm.v2.metrics.efficiency`, which scores the 20-case GM)

- **Case k**: the largest `problem_size` of the frozen 20-case set of the operator's representative dtype (ties: last).
  For all 45 operators it equals the representative case of `representative_dtypes.yaml`.
- **valid_i**: round i passed all 20 frozen cases under its **adopted** evaluation revision (`evaluation_revision` of
  the round in `trajectory.json`). Other revisions on disk are kept in `other_revisions_latency_k_ms`.
- **SOL_i = T_SOL,k / T_i,k**; values above 1 are kept.
- **C_i**: cumulative logical input + output tokens of rounds 1..i, failed rounds included.
- **BestSOL(B) = max({SOL_i : C_i ≤ B, valid_i} ∪ {0})**; per track the arithmetic mean over the eligible operators
  (45, or 36 for Trn2 NKI); operators without a valid round count 0.
- **Success@5**: share of eligible operators with at least one valid round. **Valid Rate**: valid rounds / all rounds.
- **SOL per million tokens** = mean BestSOL@5 / (mean tokens per operator / 10^6).
- **USD**: provider-reported usage × the frozen list-price snapshot (`api_pricing.yaml`, sha256 8eeac6c2…). This is
  an estimate, not the billed amount.

## Files

| file | content |
|---|---|
| `llm_rounds.csv` | one row per round (3,510): identity, status/validity, case k, T_SOL,k, T_i,k, SOL_i, adopted and other revisions, tokens (input/output/reasoning, cumulative, lower-bound flag), USD (exact / lower bound / status, cumulative), transport attempts and unknown charges, API elapsed (all attempts / successful attempt), evaluation elapsed, worker wall, device-lock wait, timestamps |
| `llm_summary_16.csv` | 8 device/DSL × 2 models |
| `figures/fig_llm_bestsol_budget.{pdf,svg}`, `previews/…png` | mean BestSOL(B) against the actual cumulative token budget per operator, 8 panels × (GPT, Claude); 6.30 × 3.05 in, fonts ≥ 7 pt |
| `appendix/llm_per_operator.csv` | per device/DSL/model/operator: valid rounds, first valid and best round, best latency and SOL, tokens, USD, API time, round statuses, human SOL |
| `appendix/llm_failure_breakdown.csv` | round statuses per track, max-token truncations, unknown-charge transports, reviewed rounds |
| `appendix/llm_cost_tokens_time.csv` | tokens, USD (exact or lower bound), API time, evaluation time, device-lock wait per track |
| `appendix/llm_campaign_wall_time.csv` | per campaign: first request start to last request/evaluation end |
| `appendix/llm_cross_device_common_ops.csv` | the 16-row summary restricted to the 36 operators common to all devices |
| `appendix/human_kernel_comparison.csv` | per device/DSL/operator: human-kernel latency at case k, its SOL, and the best GPT/Claude latency and SOL |
| `appendix/human_dev_time_template.csv` | per device/DSL/operator; development time left missing (no reliable record in the repository) |
| `appendix/llm_bestsol_curves.json` | exact step functions plotted in the figure |
| `llm_rq3_manifest.json` | inputs with hashes and commits, definitions, figure layout check |

## Data integrity (all asserted by the script)

- 702 trajectories, 3,510 rounds (B200 1,350, GH200 1,350, MI300X 450, Trn2 360), 5 rounds and 1 generation each;
  every track covers exactly its eligible operators.
- Every trajectory: formal Base, complete, scoring binding = the frozen table's profile and arithmetic modes, price
  snapshot = the frozen file; task dtype and the 20 case ids = the frozen case set.
- Adopted evaluation: status and valid-case count of `evaluation.json` match the round. 21 GH200 rounds list two or
  three revisions (19 adopt `eval_0002`; 2 keep `eval_0001` with a post-campaign re-measurement marked "not adopted").
  Two further GH200 rounds hold an `eval_0002` that the trajectory does not register (cuTile/claude flash_decode r5,
  cuTile/gpt weight_dequant r5); the metric follows the trajectory (`eval_0001`); case-k latencies differ by 1.8 % and
  0.03 %.
- 2,941 valid rounds, all with a case-k latency; 119 valid rounds have SOL > 1.
- 27 transport attempts failed after sending with an unknown charge (B200 12, GH200 9, MI300X 6, Trn2 0): their
  tokens and USD are unknown, so 67 rounds have a cumulative token count that is a lower bound and those tracks report
  USD as a lower bound.

## Comparison limits

1. **SOL reference.** RQ3 uses the frozen LLM-v2 empirical scoring tables (per-device PR #323 profiles, arithmetic
   modes rev 2). The benchmark-track figures use the hybrid reference (published dense rate for
   `matmul_fp32_fp16_fp8`, H1; compulsory-I/O Q for RMSNorm/LayerNorm, N1). The two sets of SOL values are therefore not
   directly comparable. Values above 1 come from the empirical library-GEMM rate (matmul FP16 on B200/GH200), the
   three-pass Q of RMSNorm/LayerNorm, and copy-probe bandwidth on memory-only targets (notably MI300X element-wise
   operators).
2. **Timing harness.** LLM candidates are timed by the v2 evaluator (1 warmup, 3 timed runs per case; CUDA graph and
   253 MB / 120 MB flush on B200 / GH200; eager with 512 MB flush on MI300X; Trn2 = Neuron runtime busy-sum per call,
   no flush). Human kernels come from the benchmark CSVs (formal autotuned runs of `run_bench.py`, different repeat
   counts and protocols; Trn2 = formal default `nki_ms`). The human comparison is indicative, not a same-harness
   measurement.
3. **MI300X human Triton kernels** reach a mean SOL of 0.41 at case k, and the better of the two LLM kernels exceeds
   them on all 45 operators; the human kernels are the same Triton sources as on the NVIDIA devices, run on ROCm.
4. **Trn2** has 36 eligible operators; cross-device statements use `llm_cross_device_common_ops.csv` (36 operators)
   and compare SOL proximity, never latency.
5. **Tokens.** Claude reports thinking inside its output tokens (reasoning tokens left blank); GPT reports reasoning
   tokens separately (included in output). Unknown-charge retries make some token totals lower bounds.
6. **Time.** `api_elapsed_s` is the provider request time (all attempts of the round); evaluation time is the
   evaluator's elapsed / worker wall time; device-lock wait is queueing behind other evaluations. Campaign wall time
   spans the first request to the last event and includes scheduler pauses, review waits and the concurrency of the
   six (or two) tracks, so it is not a per-track cost.
7. **Single case.** The paper metric scores one shape (case k); the framework's 20-case geometric mean is a different
   quantity and is kept only as `latency_geomean20_ms`.
8. **Human development time** is not recorded in the repository and is left missing.
