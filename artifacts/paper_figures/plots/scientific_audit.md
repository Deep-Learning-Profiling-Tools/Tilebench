# Scientific audit of the mechanisms in Figure 3 and Figure A5

This audit covers every mechanism the figures state. For each one it answers the same eight questions:
1. the measured performance difference;
2. the formal CSV values (`torch_ms`, `dsl_ms` at the profiled `case_id_v2`, autotune mode);
3. the supporting counters or ISA;
4. whether the counters are directly comparable;
5. whether different implementations or autotune winners affect the result;
6. whether a controlled experiment exists;
7. whether the claim is qualified;
8. whether the figure separates observations from hypotheses.

The counter values come from `combined/figure_evidence.csv` (evidence IDs in the manifests).

## Figure 3A / A5 B: matrix operand delivery (matmul FP32, M = N = 4096, K = 20480; case `357a02eb8c…`)

**1. Measured difference.**
- cuTile goes from 1.035× on B200 to 0.613× on GH200.
- Triton goes from 0.670× to 0.883×, and is at 0.082× on MI300X.
- The ranking of cuTile and Triton reverses between B200 and GH200.

**2. Formal CSV values (ms).**

| device | torch | Triton | cuTile |
|---|---|---|---|
| B200 | 0.9466 | 1.413 | 0.9145 |
| GH200 | 1.5731 | 1.7811 | 2.566 |
| MI300X | 2.0873 | 25.4433 | – |

**3. Counters and ISA.**
- **TMA bytes:** B200 Triton 21.5 GB, cuTile 10.7 GB; GH200 21.5 GB for both.
- **Dynamic warp-level STS:**
  - GH200 cuTile: 21.23 M, against 20.97 M WGMMA, plus 20.97 M LDSM.
  - GH200 Triton: 0.26 M.
  - B200: 0.13 M for both DSLs.
- **MI300X static ISA:** the descriptor kernel has 96 scalar `global_load_dword`, 64 `global_store_dword` and 96
  MFMA; MFMA utilization is 4.5% (rocprof-compute).
- **A5 B:** GH200 STS per WGMMA is 1.01 (FP32), 0.00 (FP16) and 16.05 (FP8, STS.U8), in one `matmul_kernel` launch.
  For FP8, the formal speedup is cuTile 0.349× against Triton 1.064×.

**4. Comparable?** Yes for B200 vs. GH200: both are NCU dynamic counts summed over the same `run()` and the problem
size is identical. MI300X static counts are shown as text only.

**5. Implementation or autotune effects?** Yes.
- The Triton harness transposes B outside the timed region (`_bt_cache`), while cuTile loads B as [K, N].
- cuTile tiles differ: 256×256×64 on B200, 128×128×32 on GH200.
- cuda-tile versions differ: 1.3.0 vs 1.5.0.

**6. Controlled experiment?** Only on MI300X: the descriptor-versus-pointer diagnostic (A5 B).
- FP32: same tile and stages, but a masked `tl.store` instead of the descriptor store. 25.1 → 7.5 ms.
- FP16: in addition, `num_stages` drops from 3 to 2. 15.2 → 2.3 ms.
- Neither is single-factor. On NVIDIA there is none.

**7. Qualified?** Yes. The text says "consistent with re-laying out the B operand", names the harness asymmetry, and
describes the MI300X diagnostic as showing the potential of an alternative implementation.

**8. Observation vs. hypothesis.** Counts and bytes are stated as measurements. The re-layout is stated as an
interpretation, with its confounders listed under the row.

## Figure 3B / A5 A: indexing overhead (destindex INT8, case `79fc14733c…`)

**1. Measured difference.**
- Triton: 4.978× (B200), 4.468× (GH200), 2.168× (MI300X).
- cuTile: 1.692× (B200), 1.589× (GH200).

**2. Formal CSV values (ms).**

| device | torch | Triton | cuTile |
|---|---|---|---|
| B200 | 0.1812 | 0.0364 | 0.1071 |
| GH200 | 0.1939 | 0.0434 | 0.1220 |
| MI300X | 0.1431 | 0.0660 | – |

**3. Counters and ISA.**
- Executed instructions: Triton 5.21 M on both devices; cuTile 90.3 M (B200) and 92.9 M (GH200).
- Stores: Triton 0.128 M STG.E.128 against cuTile 2.048 M STG.E.U8, with identical store sectors (2.05 M) on both
  devices.
- MI300X static ISA: 4 per-lane `buffer_store_byte` and no vector store.

**4. Comparable?** Yes between DSLs and between NVIDIA devices (dynamic, same `run()`). MI300X is static and shown as
text.

**5. Implementation or autotune effects?** The cuTile instruction expansion has the same size on both NVIDIA devices,
so it is not an architecture effect. The PyTorch baseline (ATen index_copy) differs by vendor, which affects the MI300X
speedup.

**6. Controlled experiment?** No.

**7. Qualified?** Yes. The text says "uses 8-bit instead of 128-bit stores" (a measurement) and states that the
baseline differs by vendor.

**8. Observation vs. hypothesis.** Yes. Instruction and store counts are observations, and no causal share of the
latency is claimed.

## Figure 3C / A5 C: memory access and latency hiding (1d_conv FP16, case `4b8e7fcaf4…`)

**1. Measured difference.**
- Triton: 0.516× (B200), 0.515× (GH200), 1.483× (MI300X).
- TileLang: 0.077× (B200), 0.104× (GH200).

**2. Formal CSV values (ms).**

| device | torch | Triton | TileLang |
|---|---|---|---|
| B200 | 1.2986 | 2.5153 | 16.8168 |
| GH200 | 1.7006 | 3.3044 | 16.4305 |
| MI300X | 2.4375 | 1.6434 | – |

**3. Counters.** All values are TileLang against Triton.

| counter | B200 | GH200 |
|---|---|---|
| load requests | 62.9 M vs 62.9 M | 188.7 M vs 94.4 M |
| L1 load sectors | 1,091 M vs 147 M | 2,181 M vs 210 M |
| DRAM read | 0.67 GB vs 0.67 GB | 0.67 GB vs 0.67 GB |
| achieved occupancy (theoretical) | 6.2% (18.75%) vs 24.6% (25%) | 18.7% (18.75%) vs 24.8% (25%) |
| issue active | 3.9% vs 57.8% | 22.4% vs 61.5% |

- All loads are 16-bit: LDG.E.U16 for Triton, LDG.E.U16.CONSTANT for TileLang.
- TileLang's L1 hit rate is 92–97%.
- On MI300X, PyTorch runs MIOpen implicit GEMM plus 3 transpose kernels (kernel trace). Triton uses
  `buffer_load_ushort` (static ISA).

**4. Comparable?** Yes on NVIDIA: dynamic counts in the same `run()`, with the same load width. The text no longer
equates sectors per request with coalescing in general. It states that the extra sectors are L1 traffic with
unchanged DRAM reads.

**5. Implementation or autotune effects?** Yes.
- TileLang uses a different kernel body and configuration on Hopper (64×16 vs 128×64 blocks).
- The MI300X change in Triton's speedup comes with a different PyTorch path.

**6. Controlled experiment?** No.

**7. Qualified?** Yes. "On B200, TileLang achieves 6.2% occupancy against a theoretical limit of 18.75%" is a
measurement, shown as achieved bars over grey theoretical bars in both Figure 3C (with achieved / theoretical labels)
and A5. The text makes no claim about why B200 TileLang falls short of its theoretical occupancy, and the MI300X
case is labelled a baseline effect.

**8. Observation vs. hypothesis.** Yes.

## A5 A: instruction expansion (weight_dequant, cross_entropy, moe_topk_gating, flash_decode)

**1. Measured difference.** cuTile speedups are below Triton's in every case shown:

| operator | B200 Triton | B200 cuTile | GH200 Triton | GH200 cuTile |
|---|---|---|---|---|
| weight_dequant BF16 | 19.5× | 7.08× | 12.1× | 8.45× |
| cross_entropy FP16 | 2.50× | 1.62× | 1.55× | 1.01× |
| moe_topk_gating FP16 | 9.60× | 4.38× | 7.94× | 4.11× |
| flash_decode FP32 | 1.044× | 0.281× | 1.013× | 0.273× |

**2. Formal CSV values (ms).** Example: flash_decode on B200 is torch 0.0471, Triton 0.0451, cuTile 0.1676. All other
values are in `benchmark_cases_normalized.csv.gz` at the profiled `case_id_v2`.

**3. Counters.** `smsp__inst_executed.sum` ratios to Triton on the same device:

| operator | B200 | GH200 |
|---|---|---|
| weight_dequant | 6.2 | 1.9† |
| cross_entropy | 3.5 | 2.5 |
| moe_topk_gating | 6.6 | 2.7 |
| flash_decode (16 CTAs) | 2.1 | 3.9 |

**4. Comparable?** Yes, within a device, using the same counter.

**5. Implementation or autotune effects?** Yes for weight_dequant on GH200 (†): Triton's autotuned configuration
differs and inflates its count. It is marked as a confounder. The GH200 moe_topk_gating Triton winner uses 64
threads per program.

**6. Controlled experiment?** No.

**7. Qualified?** Yes. The panel shows instruction counts, not latency shares.

**8. Observation vs. hypothesis.** Yes. For flash_decode, no claim is made about long-scoreboard stalls; on GH200
cuTile's ratio is not higher than Triton's.

## A5 C: MI300X vector_add load cache policy (FP32, case `67ec1efc98…`)

**1. Measured difference.** Triton reaches 0.636× (MI300X) against 1.015× (B200) and 1.007× (GH200).

**2. Formal CSV values (ms).** MI300X: torch 0.0610, Triton 0.0959.

**3. Evidence.**
- **ISA:** Triton's `.cg` load modifier emits `sc0 nt` on the loads. ATen's loads are `nt` (ATT trace of the PyTorch
  kernel). The formal Triton kernel has no `nt` loads (static ISA).
- **Diagnostic latencies (µs):**

  | flush protocol | PyTorch | Triton default loads | Triton `.cg` loads |
  |---|---|---|---|
  | 512 MiB write flush (formal) | 61.3 | 98.9 | 59.3 |
  | read flush | 52.4 | 57.7 | 51.2 |
  | no flush | 51.7 | 57.2 | 50.8 |

- **Store modifiers** (`.cs` → `sc0 nt`, `.wt` → `sc0 sc1`, `.cg` → unchanged) all measured 97.6–98.9 µs.

**4. Comparable?** Yes, within one diagnostic protocol (2 warmup and 10 timed runs, block repeated twice). The values
are never compared with formal latency.

**5. Implementation or autotune effects?**
- The diagnostic uses a standalone kernel (BLOCK 2048, 4 warps), not `impl_triton.py`'s autotuned winner.
- The launch-configuration sweep shows 97–105 µs for every configuration with default loads, so the gap is not a
  configuration effect.

**6. Controlled experiment?** Yes. The load modifier and the flush protocol are varied one at a time on the same kernel.

**7. Qualified?** Yes. The gap depends on the formal write flush (1.6×, against 1.1× without it). The panel and
caption say so and label the latencies diagnostic.

**8. Observation vs. hypothesis.** Yes.

**Correction against the first draft.** The first draft plotted the cache-modifier ablation with a 131.8 µs "default"
bar and the note "st.cg has no ISA effect" next to `st.cs` bars. The 131.8 µs value is not reproduced by the
ISA-identical `.cg`-store run (98.9 µs) or by the sweep (97–105 µs). It is now excluded as run-to-run variation and
documented in the A5 manifest.

## Claims removed or weakened in this revision

- **Figure 3 cases.**
  - The six-case text matrix is replaced by three mechanisms.
  - flash_decode, block_sparse_attention and vector_add are moved to A5, A3 and the supporting evidence rows.
- **destindex.**
  - "Store sectors/request: C 1.0, T 16" is replaced by store instruction counts and widths, with equal store sectors.
  - This avoids reading sectors per request as a coalescing score.
- **1d_conv.**
  - "Load sectors/request 17.3 vs 2.3" is replaced by request, sector and DRAM byte ratios at identical 16-bit load
    width.
  - The text now states that the extra sectors are L1 traffic.
- **MI300X GEMM.** It is described as a multi-factor diagnostic, with the FP16 bar hatched for the `num_stages` change.
- **Figure 4.** The "within 5%" lines are removed from the figure. The caption states that winners are numerical, not
  significant. Histogramming is marked as an algorithmic difference.
