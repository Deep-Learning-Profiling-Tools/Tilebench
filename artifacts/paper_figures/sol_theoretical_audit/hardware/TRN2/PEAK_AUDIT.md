# Trainium2 (trn2.3xlarge) hardware-theoretical peak audit

Status: audit candidate, not frozen. Machine-readable values: `hardware_theoretical_peaks.json`; checks: `validation.json`.
This audit ran no benchmark, profiler capture or calibration. It reads AWS documentation (Neuron 2.32.0, retrieved
2026-10-10), the hardware constants that AWS Neuron Explorer 2.32 embeds in the existing TileBench Trn2 profiles, and
existing TileBench metadata. The empirical calibration `Trn2-20261010T012011Z-94406dcb` is used only for comparison
and is unchanged.

SOL model: `T_SOL,arch = max(F / P_arch[mode], Q / BW_arch)`, all values **per logical NeuronCore under LNC=2
(= 2 physical NeuronCore-v3, one HBM stack)**. Conventions: 1 multiply-add = 2 FLOP; decimal prefixes; HBM traffic =
bytes read + bytes written; dense only.

## 1. Result table (per logical NeuronCore)

| Mode | Engine / instruction | P_arch or BW_arch | Status | Empirical (reference) | Emp / arch |
|---|---|---|---|---|---|
| `bf16_mma` | TensorE `nc_matmul` bf16 → fp32 PSUM | **157.29 TFLOP/s** | verified | 151.49 | 96.3 % |
| `fp16_mma` | TensorE `nc_matmul` fp16 → fp32 PSUM | **157.29 TFLOP/s** | verified | 146.79 | 93.3 % |
| `tf32_mma` | TensorE `nc_matmul` tfloat32 | **157.29 TFLOP/s** | derived_with_assumptions | 121.44 (compiler auto-cast) | 77.2 % |
| `fp32_tensor_mma` | TensorE `nc_matmul` float32 (1/4 rate) | **39.32 TFLOP/s** | verified | 36.93 | 93.9 % |
| `fp8_e4m3_legacy_mma` | TensorE, Neuron legacy E4M3, `double_row` | **314.57 TFLOP/s** | verified | not measured | – |
| `fp8_e5m2_mma` | TensorE, E5M2, `double_row` | **314.57 TFLOP/s** | verified | not measured | – |
| `fp8_e4m3fn_mma` | OCP E4M3FN | – | **unsupported** (NeuronCore-v4+) | host fallback | – |
| `int8_mma` | int8 TensorE | – | **unsupported** | host fallback | – |
| `fp32_vector` | VectorE/ScalarE/GpSimd FMA | – (candidates below) | **unresolved** | 0.747 (3 engines) | – |
| `bf16_vector` | same, bf16 data | – (candidates below) | **unresolved** | 0.861 | – |
| `fp16_vector` | same, fp16 data | – (candidates below) | **unresolved** | 0.860 | – |
| `hbm_bandwidth` | 1 HBM stack (24 GB bank) | **716 GB/s** (alt. 725) | derived_with_assumptions | 672.27 | 93.9 % (92.7 % of 725) |

Non-matrix engine bounds (per logical core; FMA = 2 FLOP per element):

| Engine | Width × clock | fp32 data | bf16/fp16 data | Status |
|---|---|---|---|---|
| VectorE | 256 (fp32) / 512 (bf16/fp16, 4× perf mode) elem/cycle × 0.96 GHz | 0.983 TFLOP/s (0.492 if 128 elem/cycle) | 1.966 TFLOP/s (0.983 in 2× mode) | derived_with_assumptions |
| ScalarE | 128 elem/cycle × 1.2 GHz | 0.614 | 0.614 | derived_with_assumptions |
| GpSimd | 128 fp32 lanes × 1.2 GHz, 1 FMA/lane/cycle assumed | 0.614 | 0.614 | unresolved |
| All three, naive sum | – | 2.212 | 3.195 (2.580 if VectorE perf mode excludes GpSimd) | unresolved |

## 2. Resource scope

| Level | Compute | Memory |
|---|---|---|
| Full Trainium2 device | 8 physical NeuronCore-v3; published 667 BF16/FP16/TF32, 1,299 FP8, 181 FP32, 2,563 sparse TFLOP/s (S1) | 96 GiB, 4 HBM stacks, 2.9 TB/s (S1; "3TB/s" S3); 128 DMA engines, 3.5 TB/s DMA (S1, S3) |
| Per physical NeuronCore-v3 | 1 each of TensorE (128×128 PE, 2.4 GHz), VectorE (0.96 GHz), ScalarE (1.2 GHz), GpSimd (1.2 GHz); 28 MiB SBUF, 2 MiB PSUM (S3) | 16 DMA engines, 435 GB/s DMA (T1); shares one HBM bank with its LNC partner (S4) |
| Per logical NeuronCore, LNC=2 | 2 physical cores: 2× every engine, 56 MiB SBUF | 1 HBM bank, 24 GB (S4), 716 GB/s (T1/T2); 32 DMA engines, 870 GB/s DMA |
| **One TileBench operator** | **1 logical core = 2 physical cores** | **1 HBM stack** |

Evidence for the last row:
- **Code.** `tilebench/core/neuron_native.py` (origin/main 72d7cec6) sets `NEURON_LOGICAL_NC_CONFIG=2` (read from `neuron-ls`), `NEURON_CC_FLAGS="--target trn2 --lnc 2"` and `NEURON_RT_NUM_CORES=1`. Its timing merges the per-core copies of one execution.
- **Environment records.** The formal manifest (`final_manifest.json`, sha256 2d7b5ed7…) records trn2.3xlarge with logical_neuroncore_config 2. The calibration `environment.json` records the same `env` (`NEURON_RT_NUM_CORES=1`, LNC 2; 8 physical / 4 logical cores; 96 GB).
- **Profiler traces.** All 71 Neuron Explorer device profiles report:
  - `num_physical_cores=2`, `vnc_enabled=true`, `neuroncore_idx=2` (device 0);
  - throttle counters on physical cores nc4 and nc5;
  - instructions executed on both physical cores in every profile (minimum instruction-count ratio 0.926).
- **Kernel behavior.** An NKI kernel without `nl.program_id` still occupies both cores: it runs the full workload twice (AWS, 2026-10-06; manifest LNC note). The peak therefore counts 2 physical cores regardless of how well a kernel splits work. Splitting is a kernel-efficiency question, not a ceiling question.

## 3. Sources

AWS documentation (Neuron 2.32.0, readthedocs `latest`; html sha256 in the JSON):
- S1 Trainium2 Architecture: about-neuron/arch/neuron-hardware/trainium2.html
- S2 NeuronCore-v3 Architecture: …/neuron-core-v3.html
- S3 Trainium2 Architecture Guide for NKI: nki/guides/architecture/trainium2_arch.html
- S4 Logical NeuronCore configuration: about-neuron/arch/neuron-features/logical-neuroncore-config.html
- S5 `nki.isa.nc_matmul`
- S6 Amazon EC2 Trn2 Architecture (instance table)
- S7 Trainium/Inferentia2 (NeuronCore-v2) NKI guide, the cost models
- S8 `nki.isa.tensor_scalar`
- S9 `nki.isa.activation`
- S10 NeuronCore-v2
- S11 Trainium (Trn1)
- S12 data types
- S13 Trainium3 NKI guide

AWS tool:
- **T1:** the hardware model that Neuron Explorer 2.32.0.498 writes into every Trn2 profile. Values: `tensor_engine_num_rows/cols` 128/128; clocks TensorE 2.4, VectorE 0.96, ScalarE 1.2, GpSimd 1.2 GHz; `hbm_ddr_bandwidth` 716.0e9; `dma_ddr_bandwidth` 435.0e9; 16 DMA engines per NeuronCore. These are identical in all 71 profiles.
- **T2:** Neuron Explorer 2.32 metric terminology (the MBU, MFU and throttling definitions).

## 4. Documented facts vs derived assumptions

**Documented** (quotes in the JSON `quotes`):
- **TensorE array and clock:** a 128×128 PE systolic array (S5) at 2.4 GHz (S3 Table 10; T1 shows the same).
- **TensorE datapath and FP8 mode:** 2x128 for dense BF16/FP16 and 4x128 for dense FP8_E4/E5 (S3). In double-FP8 mode "each processing element performs two pairs of FP8 multiplications … per cycle" (S3, S5).
- **Published TensorE rates:** 79 BF16/FP16/TF32, 158 FP8 and 20 FP32 dense TFLOP/s per NeuronCore-v3; 316 sparse (S2, S3).
- **TensorE data types:** inputs float8_e4m3, float8_e5m2, bfloat16, float16, tfloat32 or float32. float8_e4m3fn only from NeuronCore-v4. fp32 PSUM accumulation (S5).
- **VectorE:** 512 BF16/FP16 elements/cycle and 256 for other types, at 0.96 GHz. The 4× / 2× performance-mode conditions are documented, and computation is always FP32 (S3).
- **ScalarE:** 128 elements/cycle at 1.2 GHz (S3). `activation` computes `op(data*scale+bias)` with no extra cost, in fp32 (S9).
- **GpSimd:** 8 × 512-bit processors = 128 fp32 lanes at 1.2 GHz (S2, S3, S7). "no rule-of-thumb" for instruction cost (S7).
- **`tensor_scalar`:** applying one or two operators costs the same. ScalarE on trn2 supports multiply then add. All math is fp32 (S8).
- **Concurrent engine access:** VectorE and GpSimd can access SBUF in parallel. VectorE's performance mode uses a bus it shares with GpSimd, under hardware arbitration (S3).
- **LNC and HBM banks:** LNC=2 gives 4 logical cores of 2 physical cores each. There are four 24 GB HBM banks, each shared by two physical cores. Under LNC=2 the two physical cores share one address space (S4).
- **Device memory and DMA:** 96 GiB at 2.9 TB/s (S1); "4 HBM stacks … 3TB/s" (S3); 46.4 TB/s for 16 chips (S6). DMA is 3.5 TB/s across 128 engines, 16 per NeuronCore (S1, S3).
- **Neuron Explorer MBU** = (read + write bytes) / time / hbm_max_bandwidth, where hbm_max_bandwidth is "the maximum HBM bandwidth of a HBM stack", the same for LNC1 and LNC2 (T2). Its value is 716.0 GB/s (T1).
- **Neuron Explorer MFU** counts a multiply-add as 2 FLOP, with tensor_engine_max_ops_per_sec = 2 × TensorE elements × clock (T2). For these profiles that is 157.29 TFLOP/s (2 physical cores).
- **Throttling** limits hardware utilization below 100% (T2). The Trainium2 NeuronCore base clock is 1.2 GHz; the engines have their own clocks (T2, T1).

**Assumptions** (each listed against its mode in the JSON):
- **TF32:** streams at the BF16/FP16 rate. It appears in the published 79 TFLOP/s figure but not in the datapath table.
- **FP32 TensorE:** the 1/4 rate comes from the published 20 TFLOP/s and the documented NeuronCore-v2 4× FP32 instruction cost.
- **VectorE fp32 width:** the documented width of 256 elements/cycle applies to a single-input `tensor_scalar`. The alternative is 128, which is what the NeuronCore-v2 cost model gives.
- **ScalarE:** retires one element per lane per cycle in steady state.
- **GpSimd:** retires one fp32 FMA per lane per cycle. This is undocumented, so GpSimd is unresolved.
- **HBM locality:** the logical core's tensors live in its own bank.
- **HBM peak:** the profiler's 716 GB/s is the hardware peak, not a derated value, and all four stacks are equal.
- **Clock:** 2.4 GHz is the maximum applicable TensorE clock. No boost clock is documented, and T1 reports 2.4 for every run.

## 5. Tensor Engine derivation

`P_arch = physical_cores_used × 128 × 128 × MAC/PE/cycle × 2 FLOP × 2.4 GHz`

| Mode | MAC/PE/cycle | Per physical core | Per logical core (2 cores) | Published per core |
|---|---|---|---|---|
| BF16 / FP16 | 1 (2x128 datapath) | 78.643 | **157.286** | 79 |
| TF32 | 1 (published rate) | 78.643 | **157.286** | 79 |
| FP32 | 1/4 | 19.661 | **39.322** | 20 |
| FP8 E4 (legacy) / E5, `double_row` | 2 (4x128 datapath) | 157.286 | **314.573** | 158 |
| FP8_E3 | 1 (S3: same as BF16) | 78.643 | 157.286 | – |
| Sparse | not applicable to dense | – | excluded | 316 |

**Independent cross-check of the datapath encoding.** On NeuronCore-v2, S7 gives 2x128 at 2.8 GHz and 92 TFLOP/s. The same formula gives 128 × 128 × 2 × 2.8 GHz = 91.75 TFLOP/s, and FP32 = 92/4 = 23, as published. AWS's own MFU denominator for our profiles is 157.29 TFLOP/s (T2/T1). Neuron Explorer's instruction-level MFU (`mfu_inst`) for the profiled fp32 matmul K20480 is 0.962 at 37.83 TFLOP/s. The tool's FP32 denominator is therefore 39.32 TFLOP/s, which means AWS's profiler independently applies the 1/4 FP32 TensorE rate. For fp16 K20480 the same metric is 0.891 at 140.07 TFLOP/s against 157.29. These profile numbers are cross-checks only; none was used to pick a bound, and all stay below their bound.

### Chip-level vs core-level discrepancy (not silently resolved)

| | BF16/FP16/TF32 | FP8 | FP32 | Sparse |
|---|---|---|---|---|
| Published chip (S1) | 667 | 1,299 | 181 | 2,563 |
| 8 × derived TensorE | 629.15 | 1,258.29 | 157.29 | – |
| Chip / 8× derived | 1.060 | 1.032 | 1.151 | – |
| TensorE clock needed to reproduce the chip figure | 2.544 GHz | 2.478 GHz | 2.762 GHz | 2.444 GHz |
| Chip − 8 × published per-core | **35** | **35** | **21** | **35** |
| Chip / 4 (per logical) | 166.75 | 324.75 | 45.25 | 640.75 |

- **H1: the chip figures use a higher clock.** Rejected. Each precision would need a different clock (2.44–2.76 GHz), and AWS documents, and the profiler reports, a single 2.4 GHz TensorE clock.
- **H2: the chip figures add the non-TensorE engines.** Consistent, but undocumented. Chip − 8 × published per-core is exactly 35 TFLOP/s for FP8, BF16 and sparse, and 21 TFLOP/s for FP32. That precision-independent additive term is what you would expect if VectorE/ScalarE/GpSimd contributions were added. The documented per-engine figures do not reproduce it: 8 × (1.0 + 1.2) = 17.6 TFLOP/s FP32.
- **H3: a different power or clock regime for the full chip.** No documentation was found. Throttling only lowers utilization.
- **Decision.** The matrix modes use the TensorE-only bound, 2 × the per-core derivation. Chip/4 is recorded but not used: it is 6.0% (BF16), 3.2% (FP8) and 15.1% (FP32) above the TensorE bound, and under H2 it counts engines that matrix modes do not represent. A strict bound for a kernel that adds vector-engine FLOPs to TensorE FLOPs is **unresolved**. No TileArena mode mixes the two.

## 6. Precision-specific limitations

- **FP8.** The legacy Neuron `float8_e4m3` (and `float8_e5m2`) run at 2× through the NKI `double_row` mode only. OCP `float8_e4m3fn` is accepted only from NeuronCore-v4 (S5). The TileBench matmul NKI raises "neuronxcc rejects F8E4M3FN before TRN3". The legacy rate is **not** assigned to E4M3FN. Note that the legacy `Trainium2.json` lists fp8_e4m3fn and int8 at 316; those are not supported on NeuronCore-v3.
- **INT8.** There is no int8 TensorE input on NeuronCore-v3 (S5, S2). NeuronCore-v2 had int8 inputs (S10/S11), but no Trn2 INT8 TOPS figure exists, so `int8_mma` is unsupported and has no value.
  - The TileArena `matmul_int8` NKI implementation converts its int8/uint8 tiles to bf16 and runs a bf16 `nc_matmul` with fp32 PSUM accumulation. That path's ceiling is `bf16_mma` (157.29). It is recorded separately as `implementation_mapping_not_a_peak` and does not establish INT8 MMA support.
- **FP32.** TensorE FP32 (`fp32_tensor_mma`, 39.32 TFLOP/s) is a different thing from FP32 vector arithmetic (section 7).
  - TileArena declares fp32 MMA operators as `mma_tf32_f32acc` (7 operator/dtype pairs). The Trn2 NKI kernels run float32 `nc_matmul` at 1/4 rate, so a TF32-mode SOL is 4× tighter than the path those kernels take. The declaration is shared and was not changed.
- **Sparse.** The sparse figures (316 per core, 2,563 per chip) are listed as references only and never used as a dense peak.

## 7. Non-matrix engines

- **Instruction.** `nisa.tensor_scalar(op0=multiply, op1=add)` on VectorE, ScalarE or GpSimd, or `nisa.activation(identity, scale, bias)` on ScalarE. Both compute a×x+b, which is 2 FLOP per element, and two operators cost the same as one (S8, S9).
- **FP32 math.** All arithmetic is done in FP32 on all three engines. BF16/FP16 change only the VectorE datapath width, through the performance mode. Trn2 has no packed half2 FMA like the GPUs' fp16x2/bf16x2.
- **VectorE fp32.** The documented width is 256 elements/cycle × 0.96 GHz → 0.983 TFLOP/s per logical core. This is the upper choice. If a single-input instruction retires only one element per partition per cycle (the NeuronCore-v2 cost model), the bound is 0.492.
- **VectorE bf16/fp16.** In the 4× performance mode (SBUF→SBUF, BF16/FP16 in and out, contiguous innermost dimension) the width is 512 → 1.966 TFLOP/s. In the 2× mode (PSUM operand or non-contiguous) it is 0.983.
- **ScalarE.** 128 × 1.2 GHz → 0.614 TFLOP/s, for any data type.
- **GpSimd.** 128 fp32 lanes × 1.2 GHz → 0.614 TFLOP/s, but only if each lane retires one FMA per cycle, which is undocumented. GpSimd `tensor_scalar` math is fp32, so 16-bit data does not widen it.
- **Concurrency.** The three engines can run concurrently (S3 relaxed the NeuronCore-v2 SBUF/PSUM restrictions). However, the VectorE performance mode borrows the VectorE/GpSimd shared bus, so the 16-bit sum is not additive.
- **TensorE as an alternative.** TensorE could apply per-partition affine maps through a diagonal-matrix matmul. Its output lands in PSUM and still needs evacuation by the other engines, so it is not counted.
- **Published engine TFLOPS are not used.** AWS's "1 TFLOPS FP32" (VectorE) and "1.2 TFLOPS FP32" (ScalarE) work out to about 1024 FLOP per engine cycle. NeuronCore-v2's 2.3 and 2.9 TFLOPS work out to about 2048 FLOP per cycle. The counting convention is undocumented and does not match the documented widths under the 1 FMA = 2 FLOP rule (ScalarE would need 7.8 FLOP per lane-cycle), so these are not used as elementwise bounds.
- **Conclusion.** The vector modes count elementwise FLOPs that a kernel may spread across all three engines; the empirical probe exercised all three. A strict bound therefore needs the combined rate, which is **unresolved**: the GpSimd FMA rate is undocumented and the shared-bus arbitration is undocumented. `value` is null. The per-engine bounds and combined candidates are in the JSON. These modes cover 54 operator/dtype pairs (43 fp32, 6 fp16, 5 bf16).

## 8. HBM bandwidth for one logical core

1. **Are the four HBM groups equal and independent?** AWS documents 4 stacks with a 2.9 TB/s total and no unequal split. The profiler uses one per-stack constant, 716 GB/s, for every Trn2 execution. Equality is assumed (A3); it is not separately documented.
2. **Can one logical core use bandwidth outside its bank?**
   - Under LNC=2 the two physical cores share one 24 GB bank and one address space (S4). The profiler's MBU uses one stack for both LNC1 and LNC2 (T2).
   - DMA engines can address any HBM on the device (S3, S7), so a kernel whose tensors were spread over several banks could exceed one stack. TileBench tensors are allocated by the runtime for the single visible logical core, and their locality is assumed (A1), not observed.
3. **Do shared DMA or memory-controller resources change the ceiling?**
   - DMA is 435 GB/s per physical core (T1; 8 × 435 = 3.48 → the documented "3.5 TB/s"; 16 engines × 27.2 GB/s). A logical core therefore has 870 GB/s of DMA, plus 307 GB/s per core of GpSimd-integrated DMA.
   - The binding limit for 2 physical cores is the stack: min(716, 870) = 716.
   - For a single physical core it is DMA: 435 GB/s. The calibration's one-core diagnostic copy reached 426.5 GB/s, 98% of that.
   - The 3.5 TB/s and 870 GB/s figures are DMA throughput and are **not** used as the HBM roofline.
4. **Is 2.9 TB/s rounded?** Yes, most likely. 4 × 716 GB/s = 2.864 TB/s rounds to 2.9. The same two-significant-figure rounding takes 8 × 435 GB/s = 3.48 TB/s to the documented 3.5 TB/s. The instance figure, 46.4 TB/s = 16 × 2.9, is derived from the rounded chip figure.
5. **Why "about 3 TB/s"?** The NKI guide (S3) rounds 2.9 to one significant figure. AWS unit spelling is also unreliable: the Trn1 page writes "820 GiB/sec" where the NKI guide writes "820 GB/s". The profiler constant is decimal.
6. **Is 725 GB/s a legitimate bound?** It is a topology-derived upper candidate (2.9 TB/s ÷ 4) that inherits the rounding.
   - **Selected: 716 GB/s, derived_with_assumptions.** This is AWS's own per-stack maximum, which its profiler applies to exactly this execution scope.
   - 725 is recorded as the alternative, 1.26% higher.
   - Both are counted as read + write bytes, matching TileArena Q. Nothing is doubled for copies.
   - Reference points: the empirical 672.3 GB/s is 93.9% of 716 (92.7% of 725). The best profiled NKI kernels reached 630.7 GB/s by bytes ÷ total_time (matrix_copy and vector_add fp32), which is 88.1% of 716.

## 9. Empirical vs theoretical

**Measured / theoretical:**

| Mode | Measured, % of bound | Notes |
|---|---|---|
| BF16 | 96.3% | |
| FP16 | 93.3% | |
| TF32 | 77.2% | |
| FP32 TensorE | 93.9% | |
| HBM | 93.9% | |
| VectorE fp32 FMA | 43% | 86% of the 128-wide alternative |
| ScalarE | 58% | |
| GpSimd | 54% | against the candidate bound |
| VectorE bf16 | 27% | against the 4× bound |

No measured value exceeds its bound.

**Explanations:**
- **Throttling.** Neuron Explorer shows the hardware utilization limit below 100% in 16 of the 71 profiled cases.
  - matmul fp16 K20480: average limit 97.1%.
  - streamk: 86–87%.
  - batched_matmul: 69–71%.
  - linear_self_attention fp32: 11%.

  The theoretical bound assumes no throttling.
- **Clock.** There is no evidence of a TensorE clock other than 2.4 GHz.
- **TensorE utilization.**
  - Each MultiplyMoving has an initiation interval of max(N, 64) cycles, with N ≤ 512.
  - LoadStationary, PSUM evacuation by VectorE/ScalarE, and DMA feed add further overhead.
  - Eager `torch.matmul` schedules come from neuronx-cc, not NKI.
- **Precision path.** TF32 was measured only through `--auto-cast matmult --auto-cast-type tf32`; a native NKI tfloat32 kernel was not measured. This is the largest gap.
- **Workload size.** Empirical GEMM throughput falls at the largest sizes (fp16 118.7 TFLOP/s at M=16384). The bound does not depend on size.
- **DMA.** One physical core is DMA-bound; a logical core is stack-bound (section 8).
- **Vector probe.** The calibration probe used two dependent chains with free size 2048 per engine. It did not demonstrate the VectorE performance mode, so the low vector percentages describe that probe, not the hardware.

The theoretical values were fixed before the empirical numbers were attached, and none was raised to match a measurement or a TileBench kernel.

## 10. Notes for the central paper-figures analysis

- **Values are per logical core.** Use `peaks.<mode>.value` (per logical core). Do not multiply them by 4 (the device) or by 8 (physical cores). `mode_name_map_to_empirical_registry` maps these modes to the arithmetic-mode registry names.
- **Computable SOL modes.**
  - `mma_fp16_f32acc` (8 pairs), `mma_bf16_f32acc` (2) and `mma_tf32_f32acc` (7, with the FP32-path caveat in section 6) have values.
  - `memory_only` (33) and `no_compute_term` (4) need only BW.
- **Not computable.**
  - The 54 vector-mode pairs: unresolved, `value` null.
  - `matmul_int8` and fp8 e4m3fn: unsupported.
- **HBM.** 716 GB/s, or 725 if the owner prefers the documentation-derived figure.

## 11. Unresolved questions / decisions for the owner

1. **Per-logical-core HBM ceiling.** 716 GB/s (selected; AWS profiler per-stack constant) vs 725 GB/s (2.9 TB/s ÷ 4). The difference is 1.26%. Bank locality of the runtime's allocations is assumed, not observed.
2. **Chip-level vs core-level TensorE.** The 667/1,299/181/2,563 chip figures exceed 8 × TensorE by a precision-independent 35 TFLOP/s (21 for FP32). The TensorE-only per-logical-core bounds are verified. What the chip figures contain, and a strict bound for mixed TensorE + vector-engine FLOPs, are unresolved.
3. **TF32.** It is not in the datapath table, and only the compiler auto-cast path was measured (77%). The bound is kept at the published BF16-equal rate.
4. **Vector modes.** The combined VectorE+ScalarE+GpSimd bound is unresolved (GpSimd FMA rate, shared bus), and so is the VectorE fp32 width (256 vs 128). The owner should choose one of:
   - (a) leave these modes without a theoretical SOL;
   - (b) adopt a candidate, for example the 3-engine naive sum: fp32 2.212, bf16/fp16 3.195 TFLOP/s;
   - (c) adopt a VectorE-only bound, which is not strict for kernels that use the other engines.
5. **Shared declarations (for the central owner, not changed here).**
   - The fp32 MMA operators are declared TF32 while the Trn2 kernels run FP32.
   - `matmul_int8` has no native int8 path; its implementation path is BF16 TensorE.
6. **Minor protocol note.** The empirical protocol's `llc_equivalent` assumes 2 × 24 MiB SBUF, but S3 says 28 MiB per NeuronCore-v3. This has no effect on the selected HBM value, because the eligible probes are ≥ 256 MiB either way. The empirical profile was not changed.

## 12. Validation and provenance

`validation.json` passes all 13 required checks:
- scope;
- two cores counted once;
- no chip or chip/4 value;
- engine and mode identified;
- FP8 legacy vs OCP;
- INT8 unsupported;
- dense only;
- units and FMA convention;
- HBM vs DMA;
- independence from benchmarks;
- conflicts documented;
- empirical files unchanged (SHA256SUMS OK; profile.json a2b810f7…; public JSON 81b0b13e…);
- no tracked file modified (only this directory is new, uncommitted).

**Provenance.**
- Worktree `~/tb_wt/_llm_trn2`, branch `exp/llm-trn2` at 15712e7e; origin/main 72d7cec6.
- Neuron stack in the profiles: runtime 2.0.79381, driver 2.120.0, Explorer 2.32.0.498, neuronx-cc 2.0.404830, nki 0.7.0b1.
- Nothing was committed, pushed or merged.
