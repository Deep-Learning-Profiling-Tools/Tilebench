# 00 — Source manifest (TileBench evaluation-revision fact package)

Generated 2026-08-17 (read-only extraction; nothing re-run, nothing modified, nothing committed).

## Inspected commit

| item | value |
|---|---|
| branch inspected | `bowen/paper/NVIDIA_Verification` |
| commit SHA | `c4587b2ce08876ce1552a651012b20a0534ea269` |
| relation to `main` | `origin/main` = `0e2fd47a9bacf3611c43944d563f3f12e8933728` (#258). The inspected branch = main + one merge commit + regenerated `Figures/*.pdf` and the figure scripts under `scripts/analysis/`. All benchmark data (`results/csv`, `results/logs`, `tilebench_run/ncu_catalogue.json`, `tilebench_run/ncu/kernel_counts.json`) is byte-identical to `main` (`git diff origin/main HEAD -- . ':(exclude)Figures'` touches only `scripts/analysis/*.py`). |
| repo | https://github.com/Deep-Learning-Profiling-Tools/Tilebench |

## Latest NCU report (primary source for NCU interpretation)

`/home/bcui2/.claude/uploads/574a6b11-bde2-48ba-b7fc-3b5220d0ff05/6b92d897-NVIDIA_Report.pdf` (uploaded as "NVIDIA_Report(2).pdf"; md5 `dd8cc1b8c79ce488b73527988adcd34e`; 37 pages; the other upload `9b3f4ccb-NVIDIA_Report.pdf` is byte-identical). Referred to below as **NVIDIA_Report(2)**. It has 37 TOC sections covering 42 operators; **not covered**: `leaky_relu`, `quantize_global`, `vector_add`. Text-only pages were extracted; pages 3, 5–8, 10–13 contain screenshots (histogramming, tl.histogram note, radix_sort header, top_k_selection, weight_dequant) which were viewed visually.

## Figure → script → data traceability (traced through the scripts, not filenames)

All `Figures/*.pdf` below are the versions committed at `c4587b2c` (regenerated after #258).

| Fig | figure file | plot-generation script (function) | input data files | mode | aggregation method (as implemented) |
|---|---|---|---|---|---|
| 2 (`fig:rq1`) | `Figures/Figure2.pdf` | `scripts/analysis/fig_evaluation.py::fig_rq1_a` | `results/csv/<op>_autotune.csv` ×45 (loaded by `scripts/analysis/bench_data.py::load_main_table`; case params merged with `benchmarks/operators/<op>/config.yaml` case_defaults; roofline via `tools/llm_codegen/roofline.py` + `data/peak_performance/B200.json`) | **autotuned** | one point per operator = geometric mean over all (dtype × case) rows of per-case speedup `torch_ms/kernel_ms` (`bench_data.per_op_mean_speedup(mode="autotune")`) |
| 3 (`fig:rq2`) | `Figures/Figure3.pdf` | `fig_evaluation.py::fig_rq2_a` | `results/csv/<op>_default.csv` + `<op>_autotune.csv` | default vs autotuned (by construction) | per (op, backend): geomean over cases of R in default mode (x) and autotune mode (y) (`bench_data.per_op_mean_R`) |
| 4 (`fig:rq3-top-perf-gaps`) | `Figures/Figure4.pdf` | `fig_evaluation.py::fig_rq3_top20` | `results/csv/<op>_autotune.csv` + `tilebench_run/ncu_catalogue.json` (case selection) | **autotuned**, NCU-profiled sweep-max case | no aggregation: per (op, dtype) the CSV row whose params equal the catalogue's `default_params_per_dtype` (the case that has an .ncu-rep; fallback max `infer_problem_size` never fires); gap = max(c/t, t/c) of single-case latencies; top 20. **Changed 2026-08-17 (this session) — previously max `infer_problem_size` with first-row tie-break, which mis-selected streamk_matmul/histogramming/top_k_selection.** |
| 5 (`fig:rq4`) | `Figures/Figure5.pdf` | `scripts/analysis/fig_llm_token.py` | `results/B200/figures/evaluation/v7/_data_llm.csv` (archived LLM aggregate; **untracked, local**) → re-based with `results/csv/<op>_default.csv` (current) and `git show 35226a956751f46892559ff0ab01aa13d08982fe:results/csv/<op>_default.csv` (Jun-15 torch baseline) → writes `Figures/_data_llm_rebased.csv` (tracked) | LLM track | per (model, backend): geomean over operators with value > 0 of TokenCost@10 (M tokens) and TokenEfficiency@10 |
| 8a (`fig:rq1ext_b`) | `Figures/Figure8a.pdf` | `fig_evaluation.py::fig_rq1_b` | same as Fig 2 | autotuned | boxplot of the 45 per-op geomeans per backend; median line = median of the 45 values |
| 8b (`fig:rq1ext_c`) | `Figures/Figure8b.pdf` | `fig_evaluation.py::fig_rq1_c` | `results/csv/<op>_autotune.csv` | **autotuned** (paper caption still says default) | per op: geomean of R over cases; bar = geomean over the ops in the category; dots = per-op values |
| 9 (`fig:rq2ext`) | `Figures/Figure9.pdf` | `fig_evaluation.py::fig_rq2_b` | `results/csv/<op>_{default,autotune}.csv` | both | per (op, backend): geomean over cases of `R_autotune[i]/R_default[i]` (index-paired); boxplot per backend; median line |
| 10–15 (`fig:ncu_top_conflict`, `fig:app-ncu-*`) | `Figures/Figure10.pdf` … `Figures/Figure15.pdf` | `scripts/analysis/fig_ncu_suite.py` | `Figures/_data_ncu_conflict.csv` (220 rows; built by `scripts/analysis/ncu_conflict_data.py` from `tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep`, primary kernel = heaviest action) | NCU winner reps | per rep C_LSU; counts by diagnosis; no averaging |
| 17 (`fig:rq4ext`) | `Figures/Figure17.pdf` | `scripts/analysis/fig_llm.py` | `results/B200/figures/evaluation/v7/_data_llm_traj.csv` (untracked, local) | LLM track | none (softmax only) |

Figure numbering follows the docstrings of the scripts (Figure16 = the retired TMA-ablation figure, not regenerated).

## Final aggregate data for RQ1 / RQ2 / RQ4

* **RQ1 and RQ2:** no aggregate CSV/JSON is materialized by the pipeline — the per-op points live only in memory inside `fig_evaluation.py`. This package materializes them by replaying the identical code path (`bench_data.load_main_table` → `per_op_mean_speedup` / `per_op_mean_R` / the `fig_rq2_b` gain loop): `03_rq1_operator_points.csv`, `04_rq2_operator_points.csv`, `02_rq_summary.json`. Suite-level numbers are derived from those per-op values (see `01_aggregation_contract.md` for the exact formula and the caveat that the suite-level geomean itself is not printed by any script).
* **RQ4:** `Figures/_data_llm_rebased.csv` (written by `fig_llm_token.py`; input to Fig 5) and `results/B200/figures/evaluation/v7/_data_llm_traj.csv` (input to Fig 17). The raw `benchmarks/llm_generated/<op>/<model>/*/run_summary.json` files no longer exist on this machine (per `fig_llm.py` docstring); the archived CSVs are the only finalized LLM artifacts.

## Bank-conflict outputs

* `Figures/_data_ncu_conflict.csv` (final, regenerated 2026-08-17 from the 220 recaptured reps; 220 rows)
* `scripts/analysis/ncu_conflict_data.py` (metric definition + thresholds), `scripts/analysis/fig_ncu_suite.py` (Figs 10–15)
* rendered: `Figures/Figure10.pdf` … `Figures/Figure15.pdf`

## LLM trajectory results

* `results/B200/figures/evaluation/v7/_data_llm_traj.csv` (1464 rows; per (op, model, backend, iter): skipped, verify_clean, speedup_vs_torch, tokens_this_iter, stop_score, frozen_at_this_iter)
* `results/B200/figures/evaluation/v7/_data_llm.csv` (180 rows = 45 ops × 2 models × 2 backends; best_speedup pre-rebase, token_cost, token_efficiency, freeze_iter)
* Both are **untracked local files** (the whole `results/B200/` tree is untracked); provenance = `tools/figures/make_evaluation_figures_v7.py::build_llm_trajectory/build_llm_aggregate` (also untracked).

## Paper source (used only to find stale text)

Untracked local copy `TileBench_EMNLP_26/` (0 files tracked in git): `latex/main.tex` inputs `src/Abstract`, `src/Introduction`, `src/Related_Work_v2`, `src/TileBench_v2`, `src/Experimental_Setup`, `src/Evaluations`, `src/Conclusions`, `src/Appendix`. Evaluation = `src/Evaluations.tex`; Appendix = `src/Appendix.tex`; metric definitions = `src/TileBench_v2.tex` §Evaluation Metrics; figure wrappers = `figure_tex/RQ1.tex`, `RQ2.tex`, `RQ4.tex`, `figure_tex/Appendix/RQ{1,2,4}_Extension.tex`. (`src/TileBench.tex`, `src/Related_Work.tex` are older duplicates not \input by main.tex.)

## Files that appear stale / must NOT be used as numerical sources

| path | why |
|---|---|
| `tilebench_run/ncu/<op>/comparison.md` (45 files) and `tilebench_run/ncu/SUMMARY.md` | excluded by instruction; not consulted |
| `results/logs/time_measurement_logs/<op>_{default,autotune}.json` (90 tracked) | **stale for 21 of 90 files** (>5 % disagreement with the CSV on the same case for bitonic_sort, histogramming, interleave, jacobi_stencil_2d, leaky_relu, matrix_copy, mul2, radix_sort, relu, reverse_array, sigmoid, top_k_selection, vector_add …). The figure pipeline reads `results/csv/*.csv`, not these JSONs → CSVs are authoritative |
| `results/B200/figures/evaluation/v7/_data_main.csv` | old main table (mid-June, pre-fix data); the current pipeline rebuilds from `results/csv` |
| `results/B200/figures/evaluation/v7/fig_*.pdf` and `TileBench_EMNLP_26/figures/v3/*`, `TileBench_EMNLP_26/figures/fig_rq3_top20_perf_gaps_{default,autotune}.pdf`, `figures/fig_ncu_*.pdf`, `figures/fig_descriptor_autotune_speedup_by_operator.pdf` | old figures the paper still \includegraphics; superseded by `Figures/Figure*.pdf` |
| `tools/figures/make_evaluation_figures*.py` (v1–v7, untracked) | old pipeline (v3 produced the submitted figures); superseded by `scripts/analysis/*` |
| `results/runtime_summary.md`, `results/aggregate/*.csv` | derived summaries of an older data state |
| `results/A100/`, `dtype_rerun_logs/`, `temp/`, `NKI_TileLang_Verification/` | unrelated/local working dirs |
| any NVIDIA_Report version other than the one above | superseded (e.g. old matmul tensor-pipe 95.6–97.2 %; old lsa 779 µs) |

## Duplicated result files — which is authoritative

| duplicate pair | authoritative | note |
|---|---|---|
| `results/csv/<op>_*.csv` vs `results/logs/time_measurement_logs/<op>_*.json` | **CSV** | JSONs stale for 21/90 (see above) |
| `results/csv/*.csv` vs `results/B200/csv/` | `results/csv/` | `results/B200/csv/` is empty (per-GPU layout deferred) |
| `results/logs/autotune_logs/<op>_autotune.json` vs `tilebench_run/ncu_catalogue.json` | consistent; catalogue is derived from the autotune logs (`tilebench_run/ncu_catalogue.py`, last rebuilt for #258) — use catalogue for winners |
| `tilebench_run/ncu/*/*.ncu-rep` (220 local, untracked) vs HF `bcui2/NCU_report/ncu_report_main/` | identical set (lsa pair re-uploaded 2026-08-16); local reps were used only for kernel names/durations in `05_ncu_case_manifest.csv` (clearly labelled) |
| `dtype_rerun_logs/torch_profile/*.ncu-rep` (5 torch diagnostic reps) | not part of the finalized 220-rep set; NOT used for numbers except where NVIDIA_Report(2) itself quotes them (e.g. lsa cuBLAS GEMM 37.6 µs) |

## Timer statistic (not a paper-level aggregate)

`core/timer.py::time_kernel_proton`: per (operator, backend, dtype, case) latency = Proton `launch` scope total GPU time ÷ `repeat`, i.e. the **arithmetic mean over 100 timed launches** after 20 warmups (every operator's `config.yaml` sets `warmup: 20, repeat: 100`; `use_cuda_graph: true`, `flush_l2: true` — L2 evicted before every warmup and every timed launch by writing a buffer of 2× device L2 = 253 MiB on B200, outside the Proton `launch` scope; the whole `run()` — all of an operator's kernels — is timed as one unit, so intra-operator producer→consumer L2 reuse is preserved). This per-case mean is the value in the CSV columns `torch_ms/triton_ms/cutile_ms` (rounded to 4 decimals). Everything above the per-case level is a geometric mean.
