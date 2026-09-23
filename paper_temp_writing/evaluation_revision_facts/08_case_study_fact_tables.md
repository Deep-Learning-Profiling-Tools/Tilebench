# 08 — Case-study fact tables (Appendix E.1 / E.2 / E.3)

Compact, fully sourced facts; no paper prose. Abbreviations for sources:

* **MAN** = `paper_temp_writing/evaluation_revision_facts/05_ncu_case_manifest.csv` (CSV/Proton latencies at the
  NCU-profiled sweep-max case, taken from `results/csv/<op>_autotune.csv`; autotune winners from
  `tilebench_run/ncu_catalogue.json`). All latencies in **ms**, all ratios are CSV ratios.
* **REP pN** = `NVIDIA_Report(2).pdf` (uploaded `6b92d897-NVIDIA_Report.pdf`, 37 pp), page N. Every NCU number below
  comes from REP text or REP screenshots only. `[metric]` = an explicit NCU metric value quoted in REP;
  `[qual]` = qualitative statement in REP (no number). Everything in REP is the collaborator's manual write-up
  (the pdf has no raw NCU exports); the `[metric]` tag therefore means "a specific NCU number is stated in the
  report", not "independently verified".
* **BC** = `Figures/_data_ncu_conflict.csv` (finalized input of Figs 10–15; `conflict_score` = C_LSU of the
  heaviest kernel; `diag` band). Priority-1 figure input, cited only for bank-conflict cross-checks.
* **CODE** = `benchmarks/operators/<op>/impl_*.py` on the inspected commit (structure only).
* **MISSING** = not stated in REP; obtaining it would require opening `tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep`,
  which this fork did not do.

Verified once for this document: none of the `.ncu-rep` files and no `comparison.md` were opened.

---

## E.1 Weight Dequantization and DestIndex

### E.1.a `weight_dequant`

| item | value | source |
|---|---|---|
| dtypes / max input | fp16, bf16, fp32; `M = N = 10240`, `TILE_SIZE = 128` (case_defaults) | catalogue via MAN; REP p13 screenshot ("M = N = 10240 / TILE_SIZE = 128") |
| PyTorch path | `row_idx = arange(M)//TILE_SIZE; col_idx = arange(N)//TILE_SIZE; scale = S[row_idx[:,None], col_idx[None,:]]; return X*scale` → ATen advanced indexing materialises a full M×N scale matrix + index tensors, multiple kernels | CODE `impl_torch.py:5-8`; REP p12 ("触发 advanced indexing，输出一个完整的 scale: shape = (M, N)") |
| Triton path | single fused flattened kernel: `x = tl.load(X+offsets, mask)`, `scale = tl.load(S + s_row*S_COLS + s_col, mask)`, multiply, `tl.store` | CODE `impl_triton.py:23-28`; REP p13 ("Triton is a single fused flattened kernel … scale loads are mostly cache-resident") |
| cuTile path | same flattened per-element scale lookup schedule, but `scale_tile = ct.gather(s_ptr, (s_row, s_col), padding_value=0)` after `ct.load` of the X tile, `ct.store` result | CODE `impl_cutile.py:28,37,41`; REP p13 ("ct.gather has a heavier lowering path, adding tile index handling, padding/bounds logic, shared-memory staging, and higher register pressure") |
| benchmark latency fp16 (torch / triton / cutile) | 1.3179 / 0.0679 / 0.1864 | MAN |
| benchmark latency bf16 | 1.3198 / 0.0677 / 0.1865 | MAN |
| benchmark latency fp32 | 1.4256 / 0.1225 / 0.1909 | MAN |
| ratios fp16: T/torch, C/torch, C/T | 19.4094×, 7.0703×, 2.7452× | MAN |
| ratios bf16 | 19.4948×, 7.0767×, 2.7548× | MAN |
| ratios fp32 | 11.6376×, 7.4678×, 1.5584× | MAN |
| Fig 2 per-op geomean T/C | 1.7298394415316054 | rq12_raw.json (Fig 2 data) |
| autotune winners | Triton fp16/bf16 `BLOCK_SIZE=2048,num_warps=4`, fp32 `BLOCK_SIZE=2048,num_warps=8`; cuTile fp16/bf16 `tile=4096,occupancy=4`, fp32 `tile=2048,occupancy=4` | MAN (catalogue) |
| kernel count (T / C) | 1 / 1 | MAN (kernel_counts.json) |
| ideal traffic (fp16/bf16) | read X 10240²×2 B ≈ 210 MB, write Y ≈ 210 MB, read S 80²×2 B ≈ 12.8 KB → "≈ 420 MB" | REP p13 screenshot `[metric-arithmetic]` |
| Torch extra traffic | full scale matrix 210 MB; two M×N int64 index grids 2×10240²×8 B ≈ 1.68 GB; scale matrix re-read by the multiply kernel; further intermediate reads/writes → "Torch 实际搬了几 GB" | REP p13 screenshot `[metric-arithmetic]` |
| total dynamic instruction counts (T, C) | MISSING (not in NVIDIA_Report(2)) | — |
| cuTile/Triton instruction ratio | MISSING | — |
| global-load instruction counts | MISSING | — |
| ALU / LSU / SM activity | `[qual]` only: cuTile "更容易表现为 ALU/resource-bound，而不是像 Triton 那样主要受 HBM bandwidth 限制"; no percentages | REP p13 |
| registers / shared memory per CTA | MISSING (REP p13 says only "higher register pressure", "shared-memory staging") | — |
| achieved occupancy | MISSING | — |
| dominant stalls | MISSING | — |
| integer SASS categories (IMAD/SHF/ISETP/LOP3/SEL) | MISSING | — |
| memory requests / sectors, `sectors/request` | MISSING | — |
| bank-conflict C_LSU (primary kernel) | cuTile fp16 0.15840551842489448 (Likely), bf16 0.16780012003372377 (Likely), fp32 0.5883694858982096 (Likely); Triton 0.0 all dtypes (No direct conflict) | BC |
| source/IR/SASS observations stated in REP | Torch: `S[row_idx[:, None], col_idx[None, :]]` "不是一个轻量级 view … 生成 row_idx / col_idx / broadcast / gather 出完整 M×N scale / 再做 X*scale" (source-level, p12); Triton: computes scale index in registers (p13); cuTile: `ct.gather` lowering adds tile-index handling, padding/bounds logic, SMEM staging, register pressure (p13). No IR or SASS listing in REP. | REP p12–13 |
| provenance split | all statements are collaborator manual analysis; the only quantitative items are the traffic arithmetic on p13; no NCU counter values are quoted for this operator | REP |

### E.1.b `destindex`

| item | value | source |
|---|---|---|
| dtypes / max input | fp16, bf16, fp32, int8; `seq_len=40960`, batch 1, nope heads 12 × dim 128, rope heads 1 × dim 64 | catalogue via MAN |
| PyTorch path | `out_nope.index_copy_(0, dest_loc, kv_nope)`; `out_rope.index_copy_(0, dest_loc, kv_rope)` (2 launches; int64 dest_loc) → generic ATen `index_copy_` kernel with OffsetCalculator + int64 indices | CODE `impl_torch.py`; REP p21–22 |
| Triton path | flat elementwise: `dst_off = (dest*head_num+head)*head_dim + d`; `v = tl.load(kv_ptr+offs, mask)`; `tl.store(out_ptr+dst_off, v, mask)` (pointer tensor + masked store; adjacent `d` contiguous → coalesced) | CODE `impl_triton.py:38-42`; REP p22 |
| cuTile path | same flat decomposition; `dest = ct.gather(dest_loc, token)`; `dst_off = ct.where(mask, dst_off, -1)`; `v = ct.load(kv_flat, …)`; `ct.scatter(out_flat, dst_off, v)` (per-element scatter) | CODE `impl_cutile.py:55-61`; REP p22 |
| latency fp16 (torch / triton / cutile) | 0.1922 / 0.0441 / 0.1112 | MAN |
| latency bf16 | 0.192 / 0.044 / 0.111 | MAN |
| latency fp32 | 0.2063 / 0.0826 / 0.1254 | MAN |
| latency int8 | 0.1812 / 0.0364 / 0.1071 | MAN |
| ratios fp16: T/torch, C/torch, C/T | 4.3583×, 1.7284×, 2.5215× | MAN |
| ratios bf16 | 4.3636×, 1.7297×, 2.5227× | MAN |
| ratios fp32 | 2.4976×, 1.6451×, 1.5182× | MAN |
| ratios int8 | 4.9780×, 1.6919×, 2.9423× | MAN |
| REP-stated gap progression | "fp32 的约 1.5×，扩大到 fp16/bf16 的约 2.5×，并在 int8 下接近 2.9×" (matches MAN) | REP p22 `[metric]` |
| Fig 2 per-op geomean T/C | 2.095543756610713 | rq12_raw.json |
| autotune winners | Triton fp16/bf16/int8 `BLOCK_SIZE=1024,num_warps=2`, fp32 `BLOCK_SIZE=1024,num_warps=8`; cuTile fp16 `nope 1024/occ4, rope 512/occ8`; bf16 & int8 `nope 1024/occ4, rope 512/occ4`; fp32 `nope 512/occ4, rope 512/occ16` | MAN |
| kernel count (T / C) | 2 / 2 (nope + rope kernels) | MAN |
| Torch NCU | Compute utilization "约为 63%", effective bandwidth "约 1.4 TB/s" → instruction-bound | REP p22 `[metric]` |
| Triton NCU | "Compute utilization 较低、DRAM utilization 达到约 78%，有效带宽约 6.0 TB/s" | REP p22 `[metric]` |
| cuTile NCU | "Compute utilization 达到约 70–76%，但有效带宽只有约 1.4–4.4 TB/s" (range across dtypes) | REP p22 `[metric]` |
| Torch vs cuTile | "cuTile … 仍比 Torch 快约 1.6–1.7×" (MAN: 1.6451–1.7297) | REP p22 |
| total dynamic instruction counts / C-over-T ratio | MISSING (REP gives none for destindex) | — |
| global-load instruction counts | MISSING | — |
| ALU / LSU / SM activity | only the Compute-utilization percentages above; no pipe breakdown | REP p22 |
| registers / SMEM | `[qual]` "shared-memory staging 和较高寄存器占用" (cuTile); no numbers | REP p22 |
| achieved occupancy | MISSING | — |
| dominant stalls | MISSING | — |
| integer SASS categories | MISSING (REP p22 states only that Triton/cuTile use int32 addressing because "所有 flat offset 小于 2^{31}" and simplify div/mod) | — |
| memory requests / sectors, `sectors/request` | MISSING | — |
| bank-conflict C_LSU | cuTile fp16 2.3925257755152853 (Mild), bf16 2.385095723201147, fp32 2.524316505556956, int8 1.836523216931623; Triton 0.0 all dtypes | BC |
| source-level observations in REP | Torch: generic OffsetCalculator + runtime token/head/dim decode + int64 index math (p21–22); Triton: `dest_loc[token]` dynamic row but contiguous `d` → pointer tensor + masked `tl.store` merges adjacent lanes (p22); cuTile: `ct.scatter` generic per-element lowering, compiler cannot recognise contiguous `d` as a vector store (p22); dtype mechanism: Triton store "一条 store 指令能够覆盖越来越多的元素" as dtype narrows, cuTile scatter "偏向逐元素 lowering" (p28–29). No IR/SASS listing. | REP |
| provenance split | REP-quoted NCU numbers: 63% / 1.4 TB/s (Torch), 78% / 6.0 TB/s (Triton), 70–76% / 1.4–4.4 TB/s (cuTile) `[metric]`; everything else `[qual]` | REP |

---

## E.2 1D, 2D, and 3D Convolution

Common facts (all three ops): PyTorch dispatches to cuDNN; Triton and cuTile use a matched implicit-GEMM
formulation (output spatial positions = GEMM M, in-channels × window = reduction) — REP p1. Triton: pointer tiles
+ `tl.load` + `tl.dot` (one `tl.dot` per kernel, no `TensorDescriptor` in any conv impl — CODE grep). cuTile:
`in_tile = ct.gather(input_flat, in_lin, …)`, `w_tile = ct.gather(weight_flat, w_lin, …)`, `ct.mma`, output via
`ct.scatter(output_flat, out_lin, …)` (CODE `impl_cutile.py` lines 98/114/117/138 (1d), 111/129/132/155 (2d),
123/143/146/171 (3d)). Kernel count 1/1 per (op, dtype) — MAN.

### E.2.a Benchmark latencies and ratios (MAN; sweep-max = NCU-profiled case)

| op / dtype | max input | torch | triton | cutile | PyTorch/Triton | PyTorch/cuTile | Triton/cuTile (C_ms/T_ms) |
|---|---|---|---|---|---|---|---|
| 1d_conv fp16 | L=2621440 (batch 1, Cin=Cout=128, k=3) | 1.2986 | 2.5153 | 4.4961 | torch 1.9369× faster (T/torch 0.5163) | torch 3.4624× faster (0.2888) | 1.7875 |
| 1d_conv fp32 | L=2621440 | 1.7185 | 3.0636 | 5.3006 | 1.7828× (0.5609) | 3.0845× (0.3242) | 1.7302 |
| 2d_conv fp16 | H=320 (Cin=Cout=128, k=3) | 0.098 | 0.5386 | 0.7815 | 5.4959× (0.1820) | 7.9745× (0.1254) | 1.4510 |
| 2d_conv fp32 | H=320 | 0.1034 | 0.6098 | 0.9153 | 5.8975× (0.1696) | 8.8520× (0.1130) | 1.5010 |
| 3d_conv fp16 | H=320, D=32 (Cin=Cout=64, k=3) | 1.4166 | 13.8689 | 35.8263 | 9.7902× (0.1021) | 25.290× (0.0395) | 2.5832 |
| 3d_conv fp32 | H=320, D=32 | 2.0821 | 18.204 | 33.165 | 8.7431× (0.1144) | 15.928× (0.0628) | 1.8219 |

REP p1–2 cross-check: "cuTile requires 1.79× and 1.73× the Triton latency for FP16 and FP32 in 1D convolution,
approximately 1.45× and 1.50× in 2D convolution, and 2.58× and 1.82× in 3D convolution" ✓; "PyTorch is about 1.9×
faster than Triton for 1D convolution, 5.5× faster for 2D convolution, and 9.8× faster for 3D convolution" (FP16) ✓.
Fig 2 per-op geomean T/C: 1d_conv 1.759773491552246, 2d_conv 1.5607790955370033, 3d_conv 2.18576787652251.

Autotune winners (MAN): 1d fp16 T `128/32/128, nw8, ns3`, C `block_bl=32,block_in=32,block_out=128,occ8`;
1d fp32 T `64/16/128, nw4, ns3`, C `block_bl=128,block_in=32,block_out=64,occ4`; 2d fp16 T `128/64/64, nw4, ns4`,
C `block_bhw=32,block_in=32,block_out=128,occ4`; 2d fp32 T `64/16/128, nw4, ns3`, C same as fp16; 3d fp16 T
`128/64/64, nw4, ns2`, C `block_bdhw=128,block_in=16,block_out=64,occ4`; 3d fp32 T `64/64/64, nw4, ns2`, C same.

### E.2.b NCU facts from REP (per bullet)

| item | 1d_conv | 2d_conv | 3d_conv | source |
|---|---|---|---|---|
| cuTile/Triton dynamic instruction ratio | ≈2.4× | ≈2.3× | ≈3.9× ("In the 3D FP16 case … 27.8 billion instructions compared with 7.2 billion") | REP p1 `[metric]` (dtype of the 1D/2D ratios not stated) |
| ALU pipeline activity | "approximately 60–74% of pipeline activity is associated with ALU execution" — stated as a family-wide range, not per op/dtype | same | same | REP p1 `[metric-range]`; per-op/dtype values MISSING |
| Tensor-pipeline activity | "only about 0.9–15%" — family-wide range | same | same | REP p1 `[metric-range]`; per-op/dtype MISSING |
| Triton MMA path | tcgen05 ("Triton uses the Blackwell tcgen05 path throughout") | tcgen05 | tcgen05 | REP p1 |
| cuTile MMA path | "falls back to the legacy HMMA path for the selected 1D and 2D tile shapes" | legacy HMMA | "cuTile reaches tcgen05 in the 3D configuration" | REP p1 (see caveat below) |
| Triton bank conflicts fp32 | ≈54.5% | ≈52.0% | ≈79.6% | REP p2 `[metric]`; BC C_LSU 54.508546722880425 / 51.92819639631114 / 79.63444844887107 ✓ |
| Triton bank conflicts fp16 | "about 3–7%" | | | REP p2; BC 5.297640151757698 / 6.772615713244212 / 3.0271325808955267 |
| cuTile bank conflicts | not stated in REP | | | BC C_LSU fp16 5.083418164532594 / 0.13225747363388274 / 2.1562258688567177; fp32 1.8398671998216394 / 1.2280709886288095 / 1.8489206006732515 (all Mild/Likely) |
| registers per thread | MISSING | MISSING | Triton "up to 255 registers per thread"; cuTile "substantially fewer registers" `[qual]` | REP p2 |
| shared memory per CTA | MISSING | MISSING | Triton "roughly 49 KB of shared memory"; cuTile MISSING | REP p2 |
| achieved occupancy | MISSING (`[qual]` "restricts resident blocks" for 3D Triton) | MISSING | MISSING | — |
| dominant stalls | "NCU also attributes substantial long-scoreboard stalls around the cuTile MMA region" (family-wide) | | | REP p1 `[qual]` |
| cuDNN kernel name / family | MISSING (REP: "cuDNN convolution kernels") | "specialized Blackwell tensor-op convolution kernel" (no exact name) | MISSING | REP p1 |
| layout-conversion vs conv-body latency | "more than 1 ms of the measured PyTorch latency is spent in layout conversion, while the convolution kernel itself takes only about 0.3 ms" | conv body "about 53 μs, compared with roughly 539 μs for the Triton kernel at the same large case"; layout-conversion share MISSING | MISSING | REP p1 `[metric]` |
| evidence indexing cost grows with dimensionality | dynamic-instruction ratio 2.4× → 2.3× → 3.9×; "2D convolution must recover (h,w) and (kh,kw), while 3D convolution must additionally recover (d,h,w) and (kd,kh,kw)"; PyTorch advantage 1.9× → 5.5× → 9.8× (FP16) | | | REP p1–2 |

Caveat (flag, not from REP): a session-side SASS check in the parent context (reading the recaptured
`1d_conv/cutile_fp32.ncu-rep`, outside this fork's allowed sources) found the current 1d_conv cuTile **fp32** winner
(`block_bl=128`) lowering to UTCHMMA (tcgen05), i.e. REP's "legacy HMMA … for the selected 1D and 2D tile shapes"
holds for 1D fp16 and 2D fp16/fp32 but may not hold for 1D fp32. Verify before writing "1D falls back" without a
dtype qualifier.

---

## E.3 Tensor-Core/TMA scheduling: `matmul_fp32_fp16_fp8` and `flash_attention`

### E.3.a `matmul_fp32_fp16_fp8` (M = N = 4096, K = 20480; catalogue via MAN)

| item | FP32 | FP16 | FP8 (e4m3fn) | source |
|---|---|---|---|---|
| latency torch / triton / cutile (ms) | 0.9466 / 1.413 / 0.9145 | 0.4645 / 0.5421 / 0.4834 | 0.3148 / 0.2476 / 0.2037 | MAN |
| Triton/cuTile (T_ms / C_ms) | 1.5451066156369602 (cuTile faster) | 1.1214315266859745 | 1.2155130093274424 | MAN / rq12_raw fig4 |
| T/torch, C/torch | 0.6699, 1.0351 | 0.8569, 0.9609 | 1.2714, 1.5454 | MAN |
| REP-stated runtime | "1.55× faster in FP32, with 0.9145 ms versus 1.4130 ms" | "1.12× faster in FP16, with 0.4834 ms versus 0.5421 ms" | "1.22× faster in FP8, with 0.2037 ms versus 0.2476 ms" | REP p2 (CSV values ✓) |
| Tensor-pipeline utilization | Triton ≈52%, cuTile ≈81% ("81% active versus Triton's 52%") | within "cuTile 79–81% vs Triton 52–69%" (per-dtype value for fp16 not itemised) | within same ranges | REP p2–3 `[metric]` |
| selected tile shapes (autotune winners) | Triton `128×128×32` (GROUP_SIZE_M 8, nw4, ns3); cuTile `256×256×64` (occ8) | Triton `256×256×64` (nw4, ns3); cuTile `256×256×64` (occ8) | Triton `256×256×128` (nw4, ns3); cuTile `256×256×128` (occ4) | MAN (catalogue); REP p2–3 states the fp32 pair "constrained to 128×128×32, whereas cuTile … 256×256×64" and "For FP16 and FP8 … Triton can use larger tiles" |
| TMA load volume | Triton 21.5 GB vs cuTile 10.7 GB | MISSING | MISSING | REP p3 `[metric]` |
| TMA store volume | MISSING | MISSING | MISSING | — |
| register usage | MISSING | MISSING | MISSING | — |
| shared memory | `[qual]` Triton "cannot accommodate the 256×256×64 FP32 tile" under "the same B200 shared-memory budget"; cuTile "compiler-managed shared-memory layout sustains 256×256×64 at one CTA per SM"; numeric bytes MISSING | MISSING | MISSING | REP p2–3 |
| occupancy | `[qual]` "These kernels deliberately operate at relatively low occupancy"; numeric MISSING | same | same | REP p3 |
| Triton TMA path | `TensorDescriptor`-based TMA loads and stores (host-side `TensorDescriptor.from_tensor` for A, B^T, C) | same | same | REP p2; CODE `impl_triton.py:4,137-152` |
| cuTile TMA path | "obtains tiled memory movement through its native tile abstraction" (`ct.load` box loads) | same | same | REP p2; CODE `impl_cutile.py:62-64` |
| Triton MMA path | Tensor-Core MMA via `tl.dot(a, b.T, acc, input_precision="tf32")` (REP: "Tensor-Core MMA"; instruction generation not itemised in REP) | tl.dot | tl.dot | REP p2; CODE `impl_triton.py:76` |
| cuTile MMA path | `ct.mma`, Tensor-Core (REP: "cuTile's Tensor-Core backend") | | | REP p2–3 |
| PyTorch selected kernel path | "cuTile and PyTorch are nearly tied … 0.9145 ms and 0.9466 ms" (kernel name not stated; TF32 enabled in impl) | "optimized nvjet 256×256 2-CTA pair-MMA implementation" | `torch._scaled_mm` "with a BF16 output and subsequently converts that result back to FP8" | REP p3; CODE `impl_torch.py:4,24,27` |
| bank-conflict C_LSU | Triton 18.774862401351232 (Severe/Moderate); cuTile 0.9766105438349497 (Likely) | Triton 0.01062243825249902; cuTile 2.3254558824521516 | Triton 0.0007063892911383464; cuTile 4.382838452170697 | BC |
| why FP32 has the largest gap | "FP32 places substantially greater pressure on the on-chip storage required for pipelined GEMM. Under the same B200 shared-memory budget, Triton's allocation model, which combines multi-stage pipeline buffers with TMA store staging, cannot accommodate the 256×256×64 FP32 tile, so its best configuration is constrained to 128×128×32 … The smaller Triton tile increases A/B re-reads and therefore nearly doubles global data movement: NCU measures 21.5 GB of TMA loads for the Triton FP32 kernel versus 10.7 GB for cuTile … keeping its Tensor pipeline at 81% active versus Triton's 52%" | | | REP p2–3 |

REP p2 also states the control-case framing: "the remaining difference cannot be explained by cuTile having TMA
while Triton does not"; overall p3: "cuTile … can keep the compute pipeline substantially busier than Triton's at
every dtype".

### E.3.b `flash_attention` (fp16 only in the finalized data; batch 4, 32 heads, head_dim 128, causal, seq_len 20480)

| item | value | source |
|---|---|---|
| latency torch / triton / cutile (ms) | 15.187 / 20.4122 / 18.6327 | MAN |
| Triton/cuTile (T_ms/C_ms) | 1.095504140570073 (cuTile faster; REP "cuTile 调优后通常比 Triton 快约 9%") | MAN; REP p25 |
| Torch vs DSL | torch faster than Triton by 1.3441× (T/torch 0.7440); than cuTile by 1.2269× (C/torch 0.8151) | MAN |
| **discrepancy flag** | REP p24: "这基本解释了 Torch 的约 1.4–1.6× 优势" — the finalized sweep-max CSV gives 1.34×/1.23×; the 1.4–1.6× may refer to smaller seq_len cases or a stale measurement. Do not quote 1.4–1.6× as the sweep-max ratio. | REP p24 vs MAN |
| Fig 2 per-op geomean T/C | 0.9413429507917843 (cuTile faster on geomean) | rq12_raw.json |
| tile shapes (autotune winners) | Triton `BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2`; cuTile `tile_m=128, tile_n=128, occupancy=16` | MAN; REP p25 ("Triton … 倾向选择 128×64，因为 128×128 会造成较高的寄存器压力；cuTile 则选择 128×128") |
| PyTorch tile / schedule | cuDNN "warp-specialized FlashAttention kernel … producer/consumer 角色 … 较大的 128×128 query tile 和更持久化的调度方式" (via `torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=…)`) | REP p24; CODE `impl_torch.py:2,5` |
| Tensor-pipeline utilization | Torch ≈79%; Triton and cuTile ≈33–36% ("Torch 的 Tensor Core pipeline 活跃率约为 79%，而两个 DSL 只有约 33%–36%") | REP p24 `[metric]` |
| registers | MISSING (REP p25 only: 128×128 "会造成较高的寄存器压力" for Triton, `[qual]`) | — |
| shared memory | cuTile "约 230 KB shared memory、每个 SM 驻留 CTA 较少"; Triton MISSING | REP p25 `[metric]` |
| TMA path | all three "都使用了 Tensor Core 和 TMA"; Triton via `TensorDescriptor.from_tensor(q/k/…)`; cuTile `ct.load` boxes | REP p24; CODE `impl_triton.py:4,109-110`, `impl_cutile.py:60,80,116` |
| MMA path | Triton `tl.dot` (QK and PV); cuTile `ct.mma` (QK and PV); REP: both "调用 tcgen05 MMA" | REP p24; CODE `impl_triton.py:58,65`, `impl_cutile.py:87,121` |
| producer/consumer behaviour of PyTorch path | warps split into producer/consumer roles so "TMA 数据搬运、softmax 和两次 MMA 可以重叠执行"; DSLs execute "QK MMA → softmax/rescale → PV MMA" serially per CTA so "softmax 期间 Tensor Core 会空闲" | REP p24 |
| Triton vs cuTile state placement | Triton register-resident tile → smaller 128×64; cuTile "SMEM-centric lowering 将较多 tile 状态放在 shared memory 中，从而避免寄存器爆炸" → 128×128 | REP p25 |
| stated reason for residual cuTile/Triton difference | "更大的 N tile 减少了 K/V 循环次数，并增加了每次 MMA 的工作量，因此 cuTile 调优后通常比 Triton 快约 9%"; "大 tile 带来的收益超过了低 occupancy 的代价" | REP p25 |
| dtype-dependence / other | small seq_len: "DSL 的 grid 较小，无法充分填满全部 SM"; converges with seq_len | REP p25 |
| bank-conflict C_LSU | Triton 20.647556887501246 (Severe/Moderate); cuTile 2.9208669984794886 (Mild) | BC |
| items requested but absent | live-register count, stall breakdown, mbarrier evidence, per-kernel TMA volume: MISSING in REP (the old paper appendix's "185 live registers / stall_long_scoreboard on mbarrier" text is not in NVIDIA_Report(2) and refers to a pre-fix kernel; do not carry it over without a source) | — |
| kernel count (T / C) | 1 / 1 | MAN |

---

## Summary of MISSING items (per case study)

* **weight_dequant:** instruction counts and C/T ratio, global-load counts, ALU/LSU/SM %, registers, SMEM,
  occupancy, stalls, integer-SASS categories, requests/sectors, sectors/request, any IR/SASS listing.
* **destindex:** instruction counts and ratio, global-load counts, pipe breakdown beyond Compute-utilization %,
  registers, SMEM, occupancy, stalls, integer-SASS categories, requests/sectors, sectors/request.
* **conv family:** per-op/dtype ALU and Tensor-pipe percentages (only family-wide 60–74% / 0.9–15% ranges),
  registers/SMEM for 1D and 2D and for cuTile in 3D, occupancy for all six, cuDNN kernel names for 1D/3D,
  layout-conversion split for 2D/3D, dtype attribution of the 1D/2D instruction ratios.
* **matmul_fp32_fp16_fp8:** TMA store volume (all dtypes), TMA load volume for fp16/fp8, register and SMEM
  bytes, numeric occupancy, per-dtype tensor-pipe values for fp16/fp8 individually (only the 79–81 / 52–69 ranges
  plus the fp32 pair 81/52).
* **flash_attention:** register counts, Triton SMEM, stall breakdown, TMA volumes; plus the 1.4–1.6× Torch-advantage
  statement conflicts with the sweep-max CSV (1.34×/1.23×).
