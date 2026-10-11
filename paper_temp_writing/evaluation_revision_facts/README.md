# Evaluation-revision fact package — README (Task 12)

Generated 2026-08-17 on branch `bowen/paper/NVIDIA_Verification` @ `c4587b2ce08876ce1552a651012b20a0534ea269` (benchmark data identical to `main` @ `0e2fd47a`). Purpose: source-grounded facts for rewriting the Evaluation section and Appendix C/D/E. **No benchmark, autotune, NCU or LLM run was executed; no result/implementation/paper file was modified.** After the package was first written, one figure-pipeline change was made at the author's request: `scripts/analysis/fig_evaluation.py::fig_rq3_top20` now selects the NCU-profiled case and `Figures/Figure4.pdf` was regenerated (committed separately on the figures branch); nothing else changed.

## Generated files

| file | task | content |
|---|---|---|
| `00_source_manifest.md` | 1 | commit SHA; figure→script→data traceability for Figs 2,3,4,5,8a,8b,9,10–15,17; RQ1/RQ2/RQ4 aggregate sources; NCU report path; bank-conflict + LLM trajectory paths; paper TeX paths; stale files; duplicate files and which is authoritative; timer statistic |
| `01_aggregation_contract.md` | 2 | exact formulas/order/code/missing-value policy for per-case latency, per-case speedup, per-op geomean, suite/category geomeans, T-vs-C ratio, roofline R, RQ2 gain, medians; LLM-track formulas incl. the zero policy actually used by Fig 5 |
| `02_rq_summary.json` | 3,4,10 | RQ1, RQ2, RQ4 aggregate values with full precision, paper rounding, source path/field, notes; category tables; example lists |
| `03_rq1_operator_points.csv` | 3 | 45 rows = the exact Fig 2 points (per-op geomean speedups, T/C ratio, winner, parity flags, torch baseline class) |
| `04_rq2_operator_points.csv` | 4 | 90 rows (op×backend): default/autotuned geomean R, gain, latency geomeans, default-in-space verdict, selection rule, winner configs |
| `05_ncu_case_manifest.csv` | 5 | 110 rows (op×dtype): NCU-profiled case, CSV latencies & ratios, kernel counts/names, winner configs, report page, plus clearly-labelled NCU profile durations and NCU/CSV ratios |
| `06_rq3_operator_diagnosis.csv` + `06_notes.md` | 6 | 45-row RQ3 diagnosis (41 columns per spec) + provenance/NA log |
| `07_rq3_figure_and_pattern_summary.md`, `07a_figure4_top20.csv` | 7 | exact Fig 4 rows; direction statement (only Triton-faster); top-10 cuTile wins; mechanism map A–E for all 45 ops with report evidence and do-not-use cases |
| `08_case_study_fact_tables.md` | 8 | E.1 weight_dequant/destindex, E.2 conv family, E.3 matmul/flash_attention fact tables with per-number sources and MISSING markers |
| `09_bank_conflict_facts.md` | 9 | C_LSU definition, thresholds, counts, top cases, matched scatter, dtype behaviour, conv fp32 values, confounder/IPC checks, figures to retain |
| `10_rq4_trajectory_summary.csv` | 10 | 40 rows (model×backend×iteration) trajectory aggregates |
| `11_paper_consistency_audit.md` | 11 | 49 line-referenced stale-text findings (A aggregation 9, B mode/paths 11, C RQ1/RQ2 numbers 11, D RQ4 5, E obsolete RQ3 10, F environment 3) + verified-correct list + unsupported-statement list |

## Authoritative sources used (in precedence order)

1. Final figure inputs: `results/csv/<op>_{default,autotune}.csv` (90 files, tracked) read through `scripts/analysis/bench_data.py::load_main_table` exactly as `scripts/analysis/fig_evaluation.py` does; `Figures/_data_llm_rebased.csv` (Fig 5, written by `scripts/analysis/fig_llm_token.py`); `Figures/_data_ncu_conflict.csv` (Figs 10–15, written by `scripts/analysis/ncu_conflict_data.py`); `results/B200/figures/evaluation/v7/_data_llm.csv` and `_data_llm_traj.csv` (Fig 5 input / Fig 17; untracked local archive — raw run_summary.json no longer exists).
2. Winner configs: `tilebench_run/ncu_catalogue.json` (derived from `results/logs/autotune_logs/<op>_autotune.json`); kernel counts/names: `tilebench_run/ncu/kernel_counts.json`.
3. NCU interpretation: `NVIDIA_Report(2).pdf` (upload `6b92d897-NVIDIA_Report.pdf`, md5 `dd8cc1b8…`, 37 pages, 42 operators). All NCU metric numbers in `06/07/08` come from its text/screenshots; kernel names and per-kernel durations in `05` were read from the existing `tilebench_run/ncu/*/*.ncu-rep` files and are labelled `NCU_PROFILE_*` (never substituted for benchmark latency).
4. Current `main` implementations (`benchmarks/operators/<op>/impl_*.py`, `core/timer.py`, `tools/llm_codegen/*`) — structure/API paths/aggregation code only.
5. Paper source (untracked `TileBench_EMNLP_26/`) — only to locate stale text.

Excluded by instruction and not consulted: `tilebench_run/ncu/*/comparison.md`, `tilebench_run/ncu/SUMMARY.md`, `results/runtime_summary.md`, `results/aggregate/`, `results/B200/figures/evaluation/v7/_data_main.csv`, all `tools/figures/make_evaluation_figures*.py`, older NVIDIA report versions.

## Headline numbers (full precision in `02_rq_summary.json`)

* RQ1 (autotuned, geomean over 45 per-op geomeans): Triton **2.0178×** (paper-round 2.02×), cuTile **1.5806×** (1.58×); medians 1.9453× / 1.5763×; faster than PyTorch 36/45 and 33/45; head-to-head Triton 37 / cuTile 8 / ties 0; within 5 %: 14, within 10 %: 17.
* RQ2: geomean gain Triton **1.1831×**, cuTile **1.2180×**; medians 1.0665× / 1.0569×; autotuned per-op R ≥ 0.8: 7/45 Triton, 5/45 cuTile (≥ 0.9: 2 / 1); autotuned below default: 4 Triton (top_k_selection 0.920, dropout, fused_activation, radix_sort), 4 cuTile (rope, bitonic_sort, fused_activation, gaussian_blur); largest gains argmax 2.675× (Triton), linear_self_attention 3.841× (cuTile).
* RQ4 (geomean over ops, zero-efficiency ops excluded as in Fig 5): TokenCost@10 0.260 / 0.331 / 0.277 / 0.441 M tokens and TokenEfficiency@10 15.83 / 10.79 / 12.90 / 7.61 for GPT-5.5+Triton / GPT-5.5+cuTile / Claude+Triton / Claude+cuTile; BestSpeedup>1: 38/45, 36/43, 38/45, 33/41 (denominator = ops with a verify-clean iteration).
* Fig 4 (regenerated with NCU-profiled cases): all 20 bars are Triton-faster (max 3.716× flash_decode/fp32, 20th bar batch_normalization/fp16 1.605×; the largest cuTile wins, top_k_selection/fp32 1.581× at k=1024 and matmul_fp32_fp16_fp8/fp32 1.545×, are below the 20th bar).
* Bank conflict (220 reps): Severe/Moderate 18 (13 T / 5 C), Mild 74, Likely 37, Confounded 4, No direct 87; top case 3d_conv/Triton/fp32 C_LSU 79.63; conv fp32 Triton values 54.5/51.9/79.6 match the report.

## Source conflicts found (reported, not resolved)

1. `results/logs/time_measurement_logs/*.json` (tracked) disagree with the CSVs for 21/90 files → CSVs are authoritative (they are what the figures read).
2. **(resolved after the package was first written)** Fig 4's case selection: `fig_rq3_top20` used max `infer_problem_size` with a first-row tie-break, which put streamk_matmul at m=1024, histogramming at num_bins=64, top_k_selection at k=16 instead of the NCU-profiled largest cases. `scripts/analysis/fig_evaluation.py` was changed to select the `ncu_catalogue.json` case per (op, dtype) and `Figures/Figure4.pdf` regenerated: the two streamk bars dropped out (true sweep-max ratios 1.04–1.06×), histogramming is now num_bins=4096 (1.78×), batched_matmul/fp32, 1d_conv/fp32, batch_normalization/fp16 entered; still all-Triton-faster. `07`, `07a`, `05` (columns *_BEFORE_fix) updated accordingly.
3. Paper metric equation for the LLM track (`G_{o,b,i} = 1/|D| Σ_d …`) vs implementation (single largest verify-clean case): the finalized data stores the single-case ratio.
4. NVIDIA_Report(2) statements vs finalized CSV: flash_attention "Torch 1.4–1.6× faster" vs sweep-max CSV 1.34× (Triton) / 1.23× (cuTile); the report's "cuTile 1D+2D conv on legacy HMMA" was written before the 1d_conv cuTile fp32 winner changed (session-side SASS check found tcgen05 for that pair) — the report is followed in the tables, the discrepancy is noted, nothing was re-derived.
5. Paper says "cuda-tile 1.3.0"; the installed and report-referenced version is `cuda-tile 1.5.0` (Triton 3.6.0, torch 2.10.0+cu130, driver 595.58.03).
6. `torch_baseline_class` was harmonized between `03` and `06` for bitonic_sort (eager_multi_kernel; alternative label correctness_or_emulation_baseline — hand-written ~300-pass eager network), histogramming and moe_topk_gating (generic_aten_kernel; alternatives vendor_library_kernel / eager_multi_kernel) — see the bracketed note in `06`.

## Fields marked MISSING (genuinely absent from the finalized artifacts)

* RQ4: compilation-failure vs verification-failure vs timeout counts separately (archived CSV only has `verify_clean`); per-dtype speedups per iteration (only the largest verify-clean case is stored).
* Task 6/8 NCU metrics not stated in NVIDIA_Report(2): 29/45 operators have all numeric NCU columns NA in `06` (per-op instruction ratios, tensor/ALU pipe %, registers, SMEM, occupancy, stall shares are stated only for the operators the report quantifies: conv family, matmul, histogramming, radix_sort, cross_entropy, flash_decode, block_sparse_attention, dequantize_rowwise, relu-int8, destindex, swiglu, sigmoid, kl_divergence, jacobi, lsa, moe, softmax, batched_matmul, rmsnorm, mean_reduction, 3d_conv Triton regs/SMEM); case-study items listed at the end of `08` (e.g. weight_dequant/destindex instruction counts, `sectors/request`, IMAD/SHF/ISETP/LOP3/SEL counts; conv per-op ALU/TC split, cuDNN kernel names for 1D/3D, layout-conversion split for 2D/3D; matmul TMA store volume, reg/SMEM bytes; flash_attention registers/SMEM/stalls). These exist only inside the `.ncu-rep` files, which were deliberately not mined for new numbers.
* Torch kernel counts for the NCU manifest: only where the report states them (`05` column `torch_kernel_count`); no finalized torch NCU set exists.
* Bank-conflict per-site classification of the current top-10 (SMEM store vs load vs MMA staging).
* Suite-level RQ1/RQ2 numbers are not printed by any script (derived here from the plotted per-op values by the documented formula).

## Sufficiency to write each part

* **RQ1** — sufficient (`02`, `03`, `01`, examples in `02.rq1.safe_examples_*`).
* **RQ2** — sufficient (`02`, `04`); note the default-in-space verdicts are import-time heuristics (81 yes / 9 no) and 8 (op, backend) pairs regress below default.
* **RQ3** — sufficient for the mechanism narrative (`07`, `06`), for the Fig 4 readout (`07a`), and for the matmul/conv/flash_decode/bsa/lsa/histogramming/radix/cross_entropy/destindex/dtype-trend paragraphs; NOT sufficient to reproduce the paper's existing matmul_int8, flash_attention, streamk-SASS and 2d_conv-"no tensor core" case studies — those must be dropped or re-derived (see `11` §E/§H).
* **RQ4** — sufficient (`02.rq4`, `10`, `01` LLM section) except separate compile/verify failure counts.
* **Appendix C (metrics)** — sufficient; the RQ4 equation must be re-stated (single-case ratio, freeze/skip charging, positive-only geomeans).
* **Appendix D (bank conflicts)** — sufficient (`09`) except per-site classification of the new top-10.
* **Appendix E (case studies)** — E.1/E.2/E.3 tables in `08` are sufficient for the report-supported claims; the missing counters above must be either omitted or obtained later from the reps.

## Paper statements that still cannot be supported (from `11` §H)

TMA-descriptor ablation subsection; matmul_int8 SASS counts/conflict-way numbers; flash_attention 43.2 % / 185 regs / 22.77 vs 17.70 ms; streamk bf16 SYNCS.PHASECHK/STTM reading; top-10 bank-conflict site classification; "register spilling for cuTile" in the 2d_conv paragraph.

## Validation checklist

1. No paper-level aggregate labelled arithmetic mean — checked (`grep -i "arithmetic mean\|macro" 0*.md 02_rq_summary.json` → only mentions are of the *timer* statistic and of stale paper text being flagged).
2. No paper-level aggregate labelled macro-average — checked (same grep).
3. Every numerical claim carries a source path — `02` fields all have `source_file`; `03/04/05/06/07a/10` have source columns; `07/08/09/11` cite file/page per number.
4. Benchmark ratios use finalized CSV/figure data — yes; NCU durations appear only in columns prefixed `NCU_PROFILE_` (`05`) and in the report-quoted per-kernel facts (`06/08`), never as benchmark latency.
5. NCU interpretations use NVIDIA_Report(2) — yes (per-page text + screenshots); `_data_ncu_conflict.csv` used only for the bank-conflict figures it generates.
6. `comparison.md` not used — confirmed by all writers (grep of outputs finds only "not used" statements).
7. Nothing re-run — no `run_bench`, autotune, `ncu`, or LLM call was issued; all Python invocations were read-only replays of the aggregation code over existing CSVs/JSONs and read-only opens of existing reps for names/durations.
8. Nothing modified — `git status` shows no modified tracked files; the only additions are this directory (untracked) and scratch files under the session scratchpad.
