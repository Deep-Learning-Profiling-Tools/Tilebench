# 09 — Bank-conflict facts for Appendix D.3.6 (read-out of the FINAL outputs only)

Inspected commit: `c4587b2ce08876ce1552a651012b20a0534ea269` (branch `bowen/paper/NVIDIA_Verification`, data identical to `origin/main` @ `0e2fd47a`; only figure files/scripts differ).

Sources (priority order used here)
1. `Figures/_data_ncu_conflict.csv` — the finalized figure input (220 rows), produced by `scripts/analysis/ncu_conflict_data.py` from the 220 reps under `tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep`; consumed by `scripts/analysis/fig_ncu_suite.py` → `Figures/Figure10.pdf … Figure15.pdf` (regenerated 2026-08-17 after PR #258, commit `c4587b2c`).
2. `scripts/analysis/ncu_conflict_data.py` (metric definition, thresholds), `scripts/analysis/fig_ncu_suite.py` (what each figure plots).
3. `NVIDIA_Report(2).pdf` p.2 (cross-check of the convolution FP32 statement).
No `.ncu-rep` file was opened; no counter was recomputed; `comparison.md` was not used.

---

## 1. Metric definition (exact, from `ncu_conflict_data.py`)

Primary kernel per (op, backend, dtype) = the action with the largest `gpu__time_duration.sum` in the rep (heaviest launch of the pipeline).

```
C_LSU = 100 * l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
            / l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
branch_eff = smsp__sass_average_branch_targets_threads_uniform.pct
             (fallback: smsp__average_branch_targets_threads_uniform.pct); None if metric absent or 0
ipc_gap    = sm__inst_issued.avg.per_cycle_active − sm__inst_executed.avg.per_cycle_active
```
`C_LSU` is stored in column `conflict_score` (percent, unbounded above 100 in principle; observed max 79.63). This matches the paper's Appendix definition of \(C_{LSU}=100\cdot B_{LSU}/W_{LSU}\) with the same two NCU metric names (`src/Appendix.tex` lines 243–266) — no wording change needed there.

## 2. Diagnosis thresholds (mutually exclusive, evaluated in this order)

| diag | rule |
|---|---|
| unavailable | `conflict_score` is None (never happens in the final CSV) |
| Confounded | `branch_eff` known and `< 98.0` |
| Severe/Moderate | `C_LSU >= 10` |
| Mild | `1 <= C_LSU < 10` |
| Likely | `0.1 <= C_LSU < 1` |
| No direct conflict | `C_LSU < 0.1` |

There is no separate "no / mild / moderate / severe" four-way split in the final pipeline; "Severe" and "Moderate" are one band (`>=10`). The paper must not describe finer thresholds than these.

## 3. Coverage

| item | value |
|---|---|
| rows in `_data_ncu_conflict.csv` | **220** (110 Triton + 110 cuTile; one row per catalogue (op, dtype) pair × backend; 45 ops) |
| rows with a valid `conflict_score` | **220** (0 `unavailable`) |
| rows with `shared_wavefronts == 0` | 0 |
| rows with `conflict_score == 0.0` | 84 (Triton 51, cuTile 33) — kernels with no LSU shared-memory conflicts recorded |
| rows with `branch_eff` available | 133 of 220 (87 kernels expose no branch-uniformity metric → cannot be flagged Confounded) |
| rows with `ipc_gap` available | 220 |

**Paper text currently says "We use NCU to profile 204 kernels"** (`src/Appendix.tex` line 209). The finalized CSV has **220** profiled (op, backend, dtype) primary kernels. Replace 204 → 220.

## 4. Diagnosis counts per backend (exactly what Figure 11 stacks)

| backend | Severe/Moderate | Mild | Likely | Confounded | No direct conflict | total |
|---|---|---|---|---|---|---|
| Triton | 13 | 28 | 13 | 2 | 54 | 110 |
| cuTile | 5 | 46 | 24 | 2 | 33 | 110 |
| **all** | **18** | **74** | **37** | **4** | **87** | **220** |

Reading: Triton has more *severe* signals (13 vs 5) but also far more clean kernels (54 vs 33); cuTile's mass sits in the Mild/Likely bands (70 of 110), consistent with its SMEM-staging data path touching shared memory in most kernels at low conflict rates.

## 5. Top-20 conflict cases (exactly the bars of Figure 10; ranked by `conflict_score`)

| rank | op | backend | dtype | primary kernel | conflict_score (C_LSU %) | branch_eff | diag |
|---|---|---|---|---|---|---|---|
| 1 | 3d_conv | triton | fp32 | conv3d_kernel | 79.63444844887107 | 99.7118 | Severe/Moderate |
| 2 | top_k_selection | triton | fp32 | block_topk_kernel | 73.96667561698848 | (n/a) | Severe/Moderate |
| 3 | matmul_int8 | cutile | int8 | matmul_kernel | 72.61898980838558 | 99.9736 | Severe/Moderate |
| 4 | 1d_conv | triton | fp32 | conv1d_kernel | 54.508546722880425 | 99.6785 | Severe/Moderate |
| 5 | linear_self_attention | triton | fp32 | z_kernel | 53.598758495210575 | 100.0 | Severe/Moderate |
| 6 | 2d_conv | triton | fp32 | conv2d_kernel | 51.92819639631114 | 99.8873 | Severe/Moderate |
| 7 | batched_matmul | cutile | fp32 | bmm_kernel | 47.87526691594043 | 99.5876 | Severe/Moderate |
| 8 | linear_self_attention | cutile | fp32 | z_kernel | 38.32120266736983 | 95.1674 | **Confounded** |
| 9 | streamk_matmul | triton | fp32 | full_tiles | 22.1412045924946 | 99.9519 | Severe/Moderate |
| 10 | flash_attention | triton | fp16 | fwd_kernel | 20.647556887501246 | 99.9914 | Severe/Moderate |
| 11 | block_sparse_attention | cutile | fp16 | block_sparse_attention_kernel | 19.252598296520713 | 98.7718 | Severe/Moderate |
| 12 | matmul_fp32_fp16_fp8 | triton | fp32 | matmul_kernel | 18.774862401351232 | 99.9903 | Severe/Moderate |
| 13 | streamk_matmul | triton | fp16 | full_tiles | 18.094194680382216 | 99.9052 | Severe/Moderate |
| 14 | softmax | cutile | fp32 | softmax_online_kernel | 17.154692282315395 | 99.9341 | Severe/Moderate |
| 15 | l2_norm | triton | bf16 | _l2_norm_fwd_kernel | 12.494302739749152 | (n/a) | Severe/Moderate |
| 16 | l2_norm | triton | fp16 | _l2_norm_fwd_kernel | 11.948245769357246 | (n/a) | Severe/Moderate |
| 17 | batched_matmul | cutile | fp16 | bmm_kernel | 11.212103861899173 | 98.0229 | Severe/Moderate |
| 18 | batch_normalization | triton | fp16 | apply_batch_norm_kernel | 11.047832588575776 | (n/a) | Severe/Moderate |
| 19 | l2_norm | triton | fp32 | _l2_norm_fwd_kernel | 10.746991599182623 | (n/a) | Severe/Moderate |
| 20 | l2_norm | cutile | fp16 | l2_norm_kernel | 9.840971044476607 | 100.0 | Mild |

(cuTile kernel names carry a `_Kt…` specialization suffix in the CSV; stems shown.)

Top-20 backend split: **13 Triton / 7 cuTile**. The paper currently says "six are cuTile kernels and fourteen are Triton kernels" and names the top case as `streamk_matmul/Triton/fp32` (`src/Appendix.tex` lines 211–220) — both are stale: the top case is now **3d_conv/Triton/fp32 (79.6%)**, and streamk_matmul/Triton/fp32 sits at rank 9 (22.1%). Note the paper's follow-on claim about the top-10 conflict *sites* (SMEM stores in four cases, loads/global-to-shared copies in four, MMA source lines in two; `streamk_matmul` `tl.load(B_ptrs…)`/`LDGSTS.E`) came from a manual NCU-Source inspection of the OLD top-10; that source-line breakdown is **MISSING for the new top-10** (no finalized artifact records it; NVIDIA_Report(2) does not enumerate conflict sites) — the paper must either drop that sentence or re-derive it.

Severe/Moderate rows in full (18): 1d_conv/T/fp32 54.51, 2d_conv/T/fp32 51.93, 3d_conv/T/fp32 79.63, batch_normalization/T/fp16 11.05, batched_matmul/C/fp16 11.21, batched_matmul/C/fp32 47.88, block_sparse_attention/C/fp16 19.25, flash_attention/T/fp16 20.65, l2_norm/T/bf16 12.49, l2_norm/T/fp16 11.95, l2_norm/T/fp32 10.75, linear_self_attention/T/fp32 53.60, matmul_fp32_fp16_fp8/T/fp32 18.77, matmul_int8/C/int8 72.62, softmax/C/fp32 17.15, streamk_matmul/T/fp16 18.09, streamk_matmul/T/fp32 22.14, top_k_selection/T/fp32 73.97.

### 5a. Top Triton cases (by C_LSU)
3d_conv/fp32 79.634; top_k_selection/fp32 73.967; 1d_conv/fp32 54.509; linear_self_attention/fp32 (z_kernel) 53.599; 2d_conv/fp32 51.928; streamk_matmul/fp32 22.141; flash_attention/fp16 20.648; matmul_fp32_fp16_fp8/fp32 18.775; streamk_matmul/fp16 18.094; l2_norm/bf16 12.494.

### 5b. Top cuTile cases (by C_LSU)
matmul_int8/int8 72.619; batched_matmul/fp32 47.875; linear_self_attention/fp32 (z_kernel) 38.321 (**Confounded**, branch_eff 95.17); block_sparse_attention/fp16 19.253; softmax/fp32 17.155; batched_matmul/fp16 11.212; l2_norm/fp16 9.841 (Mild); l2_norm/bf16 9.829 (Mild); batched_matmul/bf16 9.517 (Mild); quantize_global/fp32 7.106 (Mild).

## 6. Matched Triton-vs-cuTile scatter (Figure 13)

110 matched (op, dtype) pairs (every pair has both backends).
- cuTile score > Triton score: **46**; Triton > cuTile: **32**; exactly equal: **32** — all 32 "equal" pairs are 0.0 on both sides (kernels with no LSU shared-memory conflicts on either backend).
- Both backends `>=10` on the same pair: only **linear_self_attention/fp32** (53.60 vs 38.32).
- Triton `>=10` while cuTile `<10`: 1d_conv/fp32, 2d_conv/fp32, 3d_conv/fp32, batch_normalization/fp16, flash_attention/fp16, l2_norm/{bf16,fp16,fp32}, matmul_fp32_fp16_fp8/fp32, streamk_matmul/{fp16,fp32}, top_k_selection/fp32 (12 pairs).
- cuTile `>=10` while Triton `<10`: batched_matmul/{fp16,fp32}, block_sparse_attention/fp16, matmul_int8/int8, softmax/fp32 (5 pairs).
- Largest asymmetries (Triton − cuTile, C_LSU points): 3d_conv/fp32 +77.79 (79.63 vs 1.85); top_k_selection/fp32 +73.59 (73.97 vs 0.38); matmul_int8/int8 −65.38 (7.24 vs 72.62); 1d_conv/fp32 +52.67; 2d_conv/fp32 +50.70; batched_matmul/fp32 −43.76 (4.12 vs 47.88); streamk_matmul/fp32 +21.87; streamk_matmul/fp16 +17.89; matmul_fp32_fp16_fp8/fp32 +17.80; flash_attention/fp16 +17.73; block_sparse_attention/fp16 −17.59; linear_self_attention/fp32 +15.28.

Conclusion supported by the data: neither backend dominates (points on both sides of the diagonal), which is exactly what the current Appendix text claims for Figure 13 — that sentence can stay.

## 7. dtype-dependent behaviour (Figure 12 heatmap = per (op, backend), max over dtypes; top-25 rows shown in the figure)

Top-25 (op, backend) rows by max-over-dtype score, with per-dtype values:
3d_conv/T (fp16 3.03, fp32 79.63); top_k_selection/T (fp32 73.97); matmul_int8/C (int8 72.62); 1d_conv/T (fp16 5.30, fp32 54.51); linear_self_attention/T (fp32 53.60); 2d_conv/T (fp16 6.77, fp32 51.93); batched_matmul/C (bf16 9.52, fp16 11.21, fp32 47.88); linear_self_attention/C (fp32 38.32); streamk_matmul/T (bf16 0.05, fp16 18.09, fp32 22.14); flash_attention/T (fp16 20.65); block_sparse_attention/C (fp16 19.25); matmul_fp32_fp16_fp8/T (fp16 0.01, fp32 18.77, fp8_e4m3fn 0.00); softmax/C (fp16 6.07, fp32 17.15); l2_norm/T (bf16 12.49, fp16 11.95, fp32 10.75); batch_normalization/T (bf16 7.02, fp16 11.05, fp32 0.00); l2_norm/C (bf16 9.83, fp16 9.84, fp32 6.19); rmsnorm/T (bf16 9.04, fp16 9.77, fp32 6.78); matrix_transpose/T (bf16 1.42, fp16 1.37, fp32 8.16, int8 7.44); bitonic_sort/T (fp16 7.41, fp32 2.98); matmul_int8/T (int8 7.24); quantize_global/C (fp32 7.11); batched_matmul/T (bf16 6.92, fp16 4.66, fp32 4.12); matrix_transpose/C (bf16 6.78, fp16 6.92, fp32 5.44, int8 2.75); layernorm/C (bf16 6.09, fp16 6.37, fp32 2.82); rope/C (fp16 6.19, fp32 5.76).

FP32-vs-FP16 pattern: for 14 (op, backend) rows the fp32 score exceeds 2× the fp16 score (and is ≥1); the dominant ones are the Triton conv family (3d 3.03→79.63, 1d 5.30→54.51, 2d 6.77→51.93), batched_matmul/cuTile (11.21→47.88), matmul_fp32_fp16_fp8/Triton (0.01→18.77), softmax/cuTile (6.07→17.15), matrix_transpose/Triton (1.37→8.16). This supports the current Appendix sentence "Some FP32 cases suffer from more severe bank conflict compared to other data types" — but it is Triton-specific for the conv family and mixed elsewhere; the counter-direction (fp16 > fp32) also occurs (batch_normalization/T 11.05 vs 0.00; l2_norm/T fp16 11.95 vs fp32 10.75).

## 8. Convolution family values (all dtypes, both backends) — cross-check against NVIDIA_Report(2) p.2

| op | Triton fp16 | Triton fp32 | cuTile fp16 | cuTile fp32 |
|---|---|---|---|---|
| 1d_conv | 5.297640151757698 | **54.508546722880425** | 5.083418164532594 | 1.8398671998216394 |
| 2d_conv | 6.772615713244212 | **51.92819639631114** | 0.13225747363388274 | 1.2280709886288095 |
| 3d_conv | 3.0271325808955267 | **79.63444844887107** | 2.1562258688567177 | 1.8489206006732515 |

Report p.2: "In FP32 convolution, its [Triton's] shared-memory staging exhibits severe bank conflicts, approximately 54.5% for 1D, 52.0% for 2D, and 79.6% for 3D, compared with only about 3–7% for FP16." → **matches the finalized CSV exactly** (54.51 / 51.93 / 79.63; fp16 5.30 / 6.77 / 3.03 ⊂ 3–7%). cuTile conv kernels stay in the Mild band (0.13–5.08). These values are `C_LSU` scores of the primary kernel, not "% of runtime"; the paper should call them conflict scores.

## 9. Branch-divergence confounder (Figure 14)

Rows flagged Confounded (branch_eff < 98.0): **4** — argmax/Triton/fp16 (C 0.62, branch_eff 88.10), argmax/Triton/fp32 (C 0.81, 87.80), radix_sort/cuTile/int32 (C 0.87, 80.00), **linear_self_attention/cuTile/fp32 (C 38.32, 95.17)**. Only the last one would otherwise be Severe/Moderate; the diagnosis rule keeps it out of the severe band. 87 rows have no branch-uniformity metric and therefore can never be flagged Confounded (a limitation to state, not hide). Figure 14 plots the 133 rows with branch_eff.

## 10. IPC-gap sanity check (Figure 15)

`ipc_gap` available for all 220 rows; range −0.000359 … +0.000712 (essentially zero — issued≈executed for every kernel). Spearman rank correlation between `ipc_gap` and `conflict_score` over 220 rows = **0.205** (weak) — supports the current Appendix statement that IPC gap does not track conflict severity and is used only as a sanity check.

## 11. Figures to retain in the Appendix and their paths

All six regenerated PDFs (branch `bowen/paper/NVIDIA_Verification` @ `c4587b2c`, from the 220-row CSV above; note Figures 10, 13, 14, 15 changed at coordinate level after PR #258, 11 and 12 did not):

| paper label / current stale path in `src/Appendix.tex` | new file to use |
|---|---|
| `fig:ncu_top_conflict` — `figures/fig_ncu_top_conflict_bar.pdf` (line 233) | `Figures/Figure10.pdf` (top-20 bar) |
| `fig:app-ncu-diagnosis-stacked` — `figures/fig_ncu_diagnosis_stacked.pdf` (line 289) | `Figures/Figure11.pdf` (diagnosis stacked bar) |
| `fig:app-ncu-conflict-heatmap` — `figures/fig_ncu_top_conflict_heatmap.pdf` (line 305) | `Figures/Figure12.pdf` (top-25 op/backend × dtype heatmap) |
| `fig:app-ncu-backend-comparison` — `figures/fig_ncu_backend_comparison.pdf` (line 321) | `Figures/Figure13.pdf` (matched scatter, floor 1e-3, log axes) |
| `fig:app-ncu-branch-confounder` — `figures/fig_ncu_branch_confounder.pdf` (line 338) | `Figures/Figure14.pdf` |
| `fig:app-ncu-ipc-gap` — `figures/fig_ncu_ipc_gap.pdf` (line 353) | `Figures/Figure15.pdf` |

Retain: all six. Recommendation-neutral note: Figure 12 (heatmap) and Figure 15 (ipc gap) carry the least information for the paper's argument; both are still supported by data.

## 12. Items the paper cannot support from the finalized outputs (MISSING)

- Per-source-line conflict-site classification of the *current* top-10 (SMEM store vs load/LDGSTS vs MMA operand staging) — not in any final artifact.
- Any speedup/latency impact estimate of a given conflict score — the CSV/figures record severity only; the paper already caveats this in the definition subsection (keep that caveat).
