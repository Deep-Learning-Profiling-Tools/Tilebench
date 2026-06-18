# TileBench Triton-vs-cuTile NCU Audit (v2 — SASS/PTX-backed) — 45 operators

_Target: NVIDIA B200 (sm_100). Date: 2026-06-17. Branch: bowen/fix/EMNLP_Verification. Analysis only — no operator code modified; all suggested changes are flagged, not applied._

**Method (v2):** every operator re-audited at the **instruction level** — a per-op helper pulls correct-action NCU metrics **plus the SASS opcode mix** (`ncu --import <rep> --page source --print-source sass`), cross-referenced with `impl_torch/triton/cutile.py` + the CSV sweeps, and each medium/high finding was adversarially re-verified. **112 agents; verdicts: 43 confirmed / 21 partial / 3 refuted.** This corrects the earlier metric-only pass (which mislabeled several tensor-core/TMA conclusions). All wrong-kernel NCU captures were fixed first (re-profiled with `--kernel-name`), and the profiling harness was permanently hardened.

SASS opcode key (sm_100): **UTCHMMA** = Blackwell 5th-gen tcgen05 tensor MMA (fp16/bf16/tf32); **UTCQMMA** = tcgen05 fp8; **UTCIMMA** = tcgen05 int8; **HMMA/IMMA** = older Hopper-class tensor MMA; **UTMALDG/UTMASTG/UTMACCTL** = TMA (Tensor Memory Accelerator) bulk async copy; **LDGSTS** = Ampere cp.async; **LDSM** = load-matrix-to-shared.

## Headline: the systemic backend split (instruction-level)

The single most important finding the SASS pass establishes:

- **For matmuls, BOTH backends use the same Blackwell 5th-gen tensor cores** (`UTCHMMA`/`UTCQMMA`/`UTCIMMA`). It is **not** "one uses tensor cores, the other doesn't."
- **The real difference is the data-movement path:** **cuTile lowers `ct.load` to TMA (`UTMALDG`)**, while **Triton uses cp.async (`LDGSTS`) + `LDSM`**.
- **Consequence:** when a matmul is load-bound, cuTile's TMA keeps the tensor cores fed (high tensor-pipe %) while Triton's cp.async path, with uncoalesced `sec/req=16` loads, **starves** its (identically-issued) tensor cores. This is why Triton fp32/fp8 matmuls are far slower despite issuing the same MMA opcodes.
- **TMA is not universally better:** for small 1-D row tiles (softmax) cuTile's TMA *setup* overhead loses to Triton's plain loads; and TMA only fires for contiguous tile loads — `ct.gather`/data-dependent indices (conv im2col, stencils, sorts) lower to plain loads on both backends.

## Data-quality fix — RESOLVED + harness hardened

3 ops had captured the wrong kernel (brittle `--launch-skip 3·N` heuristic): **streamk_matmul** (read the staging-copy `vectorized_elementwise_kernel` instead of `first_wave`/`full_tiles`), **destindex** (torch argsort radix kernels), **kl_divergence** (input-gen softmax). All re-profiled with `ncu --kernel-name regex:`. **`ncu_one.py` + `ncu_driver.py` were permanently hardened** to select kernels by name from `kernel_counts.json` (`kernel_regex()` drops ATen aux kernels + strips cuTile's `_Kt…` suffix) — verified by smoke test.

## Tier 1 — Real implementation gaps (HIGH, code change warranted)

| op | who's slow | instruction-level root cause | key SASS + metrics | suggested fix (NOT applied) |
|---|---|---|---|---|
| **matmul_fp32_fp16_fp8** | **Triton** 6.2×(fp32)/2.5×(fp8)/1.24×(fp16) | Both issue tcgen05 MMA; Triton's **cp.async load path starves the tensor pipe**, cuTile's **TMA** keeps it fed | fp32: T `UTCHMMA`×8 + `LDGSTS`×80 + `LDSM`×16, `sec/req=16`, tensor%=**10**, dur 6.13ms; C `UTCHMMA`×4 + `UTMALDG`×5, tensor%=**86.8**, dur 0.98ms. fp8: T tensor%=29 vs C 77 | give Triton a TMA path (`tl.make_tensor_descriptor`); raise `BLOCK_K=32`→64/128; verify TF32/fp8 coalescing |
| **streamk_matmul** | **cuTile** ~6× (fp16/bf16); fp32 ~tied | cuTile `full_tiles` issues `UTCHMMA` but **TMA loads aren't overlapped with MMA** and **regs=255** caps occ 10.9%; Triton's pipelined cp.async+LDSM reaches tensor%=45 | fp16: T `UTCHMMA`×4 tensor%=**45.1** occ 24.7 regs 128 (`LDGSTS`×20,`LDSM`×16); C `UTCHMMA`×2 tensor%=**6.2** occ 10.9 regs 255 (`UTMALDG`). fp32: T 100×`LDGSTS` mem_sol 88 (mem-bound) vs C TMA mem_sol 12 → cuTile edges | cut cuTile `full_tiles` regs (<255 via smaller TM·TN / split-K); double-buffer `ct.load` to overlap TMA with `ct.mma`; tune `full_tiles` separately from `first_wave` |
| **2d_conv** | **cuTile** up to 5.8× (fp32), 2.5× (fp16) | im2col implicit-GEMM is **ALU-bound** (per-element index/mask math), tensor pipe idle 93–97% — MMA opcode irrelevant. fp16: cuTile lowered MMA to **legacy HMMA** (not tcgen05). fp32: **cuTile register-spills** | fp16: T `UTCHMMA`×8 tensor%=2.49 alu 67.5; C `HMMA.16816`×16 tensor%=6.9 alu 62. fp32: C `local_ld=1.7e8` dram_wr 38% (spill) vs T `LDGSTS`×48 no spill | hoist index/mask out of inner loop; im2col buffer so loads coalesce/TMA-fire; lower cuTile fp32 BLOCK/occupancy to stop spill |
| **linear_self_attention** | **both** ≪ torch; cuTile ~4× slower than Triton | **Neither uses tensor cores** — both decompose the two GEMMs into a **(D,D)=65536-CTA grid of scalar reductions** (`FADD/FMUL/FFMA/SHFL`, `sec/req=31.9`). cuTile launches 128-thread CTAs vs Triton 32 → 4× redundant uncoalesced loads | both tensor_mma=0; T `_kv_kernel` 4896µs vs C 19573µs; block_size 32 vs 128 | reformulate both stages as tiled `tl.dot`/`ct.mma` GEMMs with an M-reduction loop |
| **block_sparse_attention** | **cuTile** ~3.7× slower | Both use tcgen05 (`UTCHMMA`); cuTile has the **better load path** (TMA, `sec/req=1.0`) yet loses — **latency-bound on the serial "safe online softmax" dependency chain** (impl_cutile.py:96–166) that can't be hidden | C `UTCHMMA`×24 + `UTMALDG`×10, tensor%=5.14; T `UTCHMMA`×32 + `LDGSTS`×56 `sec/req=14.8`, tensor%=19.6 | use the plain flash update for interior (fully-unmasked) blocks; only the diagonal block needs safe-softmax |
| **weight_dequant** | **cuTile** 2.9× (fp16/bf16), 1.75× (fp32) | Elementwise (no tensor cores, correct). cuTile is **ALU-bound** on per-element `ct.gather` index math (div/mod) + `LDSM` staging; Triton stays HBM-bound | C alu **84%**, mem_sol 33%, `LDSM`×2; T tensor_mma=0, plain loads, mem_sol 71% | broadcast the block-constant scale (avoid per-element gather/index recompute) → HBM-bound |
| **histogramming** | both ≪ torch; cuTile 1.9× (autotune) | Atomic-scatter stage-1 dominates: 256-CTA cap → waves 0.14; **cuTile autotune tunes only the `occupancy` hint** (can't add CTAs) while Triton tunes BLOCK/warps | stage-1 empty SASS both, `sec/req=16`, occ ~10; reduce stage: C uses TMA `UTMALDG`×35 (5× faster but immaterial) | smem-privatized single-pass histogram; scale `num_partials` with SM count; give cuTile a grid knob |
| **radix_sort** | both ≪ torch; cuTile ~1.3× | 1-bit×32-pass design (vs cub multi-bit). cuTile scatter **spills** (`local_ld=156k`, regs 64) and lowers `ct.cumsum` to scalar-ALU; Triton's `tl.cumsum` uses `LDSM` shared scan | C `local_ld=156256` no `LDSM`; T `local_ld=0` + `LDSM`×2 | multi-bit/cub-style rewrite; cuTile: drop one cumsum, reuse buffers |
| **top_k_selection** | both 1.6–7× ≪ torch; cuTile ~1.4× | Multi-launch bitonic sort (~210 launches, O(N log²N), full HBM round-trip/step). cuTile at half Triton's occupancy (33.7 vs 63.2, regs 32 vs 20–25) | all tensor_mma=0/TMA=0; latency/launch-bound | block-local top-k reduction (O(N), one launch); raise cuTile default occupancy=32 |
| **mean_reduction** | **cuTile fp16** 3.5×; bf16 ~1.1× | Reduction (no tensor cores, correct). **cuTile fp16 register explosion**: regs=142 (vs 30 fp32 / 58 bf16 on identical code) → occ 17.7% → can't saturate HBM | C fp16 regs **142** occ 17.7 mem_sol 19; T fp16 regs 30 occ 81 | investigate fp16 reg blow-up; smaller tile / occupancy hint for fp16 |

## Tier 2 — Medium gaps, grouped by mechanism (instruction-level)

- **cuTile register spill / pressure** (verified via `local_ld>0`): `l2_norm` fp16 (`local_ld=122880`), `interleave` fp32 (`local_ld=195360`, cat/transpose materialization), `dequantize_rowwise` (whole `(1,COLS)` tile), `batch_normalization` fp32 K3.
- **cuTile occupancy/wave deficit on memory-bound ops** (same SASS class, fewer waves than Triton): `sigmoid` fp16/bf16 (waves 5.16 vs 10.31, mem_sol 40 vs 65), `swiglu` fp16/bf16 (occ 50 vs 80), `layernorm` fp16/bf16 (occ 43 vs 61), `mul2`/`matrix_copy`/`matrix_transpose`/`vector_add` **int8** (1-byte under-vectorized).
- **cuTile `ct.gather` + index-ALU instead of tiled `ct.load`** → ALU-bound on memory-bound ops: `2d_max_pooling` (alu 88 vs Triton mem_sol 97), `gaussian_blur` (alu 93–95), `argmax` fp16 (8× more load requests), `rope` (degenerate (1,1,1,64) per-head tiles).
- **TMA-vs-cp.async where it matters**: `batched_matmul` fp32 (Triton 60×`LDGSTS` `sec/req=16` mem_sol 86 starves its `UTCHMMA`×8 → cuTile TMA 1.67× faster); `matmul_int8` (Triton **legacy IMMA**×64 vs cuTile **tcgen05 UTCIMMA**×8 + TMA — autotuned cuTile 1.39× faster, but its *default* config is 1.5–1.7× slower).
- **cuTile autotune searches the wrong knob** (only `occupancy`): `l2_norm` fp32, `rmsnorm` bf16 outlier, `sigmoid`/`gaussian_blur` fp32 regressions, `histogramming`.
- **`1d_conv`** (cuTile *wins* 1.7–2.5×): Triton's 127 per-tap shifted `tl.load` is uncoalesced (`sec/req=14.8`, mem_sol 99) vs cuTile's coalesced `ct.gather` (`sec/req=2.7/4.4`) — fixable on the *Triton* side (halo-tile reuse).
- **`flash_decode`** (cuTile 3.5×): cuTile's per-iter `ct.load` of a tiny `(1,1,1,128)` tile is un-pipelined; Triton uses `num_stages=2` prefetch. Grid is only 16 CTAs (structurally starved).
- **`destindex`**: backends ~tied at the kernel level; cuTile's *default* config is ~20% better; both ≪ torch (per-token scatter geometry, Tier 3).

## Tier 3 — Slower than torch by design (NOT bugs — for the paper)

`matmul_fp32_fp16_fp8` fp16 & `batched_matmul`/`streamk_matmul` (vs **cuBLAS** persistent/warp-specialized GEMM); `flash_attention`/`flash_decode` (vs **cuDNN** fused FA); `radix_sort`/`top_k_selection`/`histogramming` (vs cub multi-bit/privatized); `destindex` (per-token scatter vs fused `index_copy_`). These are algorithm/library-baseline mismatches — the suite runs the same algorithm across backends while torch dispatches a fundamentally better vendor kernel. (Also: `bitonic_sort`/`top_k` torch baselines are *slower* host-loop references; and the fp8 torch baseline is fp32-emulated — flag in the paper.)

## Tier 4 — Verified clean / corrected (no action)

- **Genuinely clean** (non-matmul, both near roofline or tied): `3d_conv`, `batch_normalization` (K1 shared bottleneck), `dropout`, `fused_activation`, `jacobi_stencil_2d`, `leaky_relu`, `moe_topk_gating`, `quantize_global`, `relu`, `reverse_array`, `vector_add` (fp/bf), `mul2` (fp/bf), `matrix_copy`/`matrix_transpose` (fp/bf), `mean_reduction` fp32.
- **`kl_divergence` → CLEAN**: both near HBM roofline (cuTile uses TMA `UTMALDG`×16 → 78.8% vs Triton plain 73.7%); the earlier flag was the wrong-kernel artifact.
- **3 v1 findings REFUTED by the SASS pass**: `cross_entropy` fp16 (occupancy claim — Triton's default num_warps still loses but the mechanism was mis-stated; fp32 confirmed), `rmsnorm` fp16/bf16 (the "TMA/tensor-starved" labels — it's a *shared* uncoalesced-load bottleneck on both backends, `sec/req=16`, not a backend differentiator).
- **`softmax`**: cuTile *does* lower to TMA (`UTMALDG`×40) but the TMA setup overhead makes it 1.3–1.6× *slower* than Triton's plain loads on the small two-pass row tiles — a case where TMA is the wrong tool. Not a bug.

## Cross-cutting recommendations (refined)

1. **Triton matmul load path is the biggest single lever**: its cp.async (`LDGSTS`)+`LDSM` path with `sec/req=16` starves tensor cores (fp32 matmul 10% util, fp8 29%). A TMA-descriptor path (`tl.make_tensor_descriptor`) would feed the (already-issued) `UTCHMMA`/`UTCQMMA` cores — directly fixes matmul_fp32_fp16_fp8, batched_matmul fp32.
2. **cuTile autotune needs real knobs** (pipeline-depth / threads-per-row), not just the `occupancy` hint — fixes histogramming, l2_norm, sigmoid, layernorm regressions, streamk overlap.
3. **cuTile register pressure**: the fp16 reg explosion (mean_reduction 142), spills (2d_conv fp32, l2_norm, interleave, radix_sort), and the 255-reg ceiling (streamk, matmul) repeatedly cap occupancy — worth a codegen look.
4. **Two genuine "no tensor cores at all" ops**: `linear_self_attention` and `2d_conv` fp16 run the contraction on scalar pipes — reformulate as tiled `tl.dot`/`ct.mma`.
5. **TMA is not free**: for small/1-D tiles (softmax) and data-dependent gathers (stencils, sorts) it doesn't help or doesn't fire — don't force it blindly.

## Next steps

- Prioritized fix list (which `impl_*.py` first, by expected impact) for the next round.
- All findings above are described only; no operator code modified this round. Harness hardening (`ncu_one.py`/`ncu_driver.py`) is the only code change applied.
