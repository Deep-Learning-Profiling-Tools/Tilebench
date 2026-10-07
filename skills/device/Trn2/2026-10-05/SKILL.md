---
name: trn2-device-context
device: Trn2
snapshot: "2026-10-05"
kind: device_context
arch: trainium2
revised: 2026-10-05
---

# Trn2 (AWS Trainium2) Device Context (snapshot 2026-10-05)

Sourced hardware facts only, in AWS Neuron native terms (chip, NeuronCore, Tensor / Vector /
Scalar / GpSimd engines, SBUF, PSUM, DMA engines, NeuronLink). Every number carries a unit, a
scope and a source tag `[S<n>]` (see Sources). `unknown` means no allowed source states the
value. Whether NKI exposes a mechanism listed here is NOT asserted in this file; confirm it in
the NKI Reference Skill. No benchmark-derived guidance. There is NO runtime capture for this
device: `verified_on_device: false`; the Software snapshot is entirely `unknown`.

## Identity

- Chip: AWS Trainium2, "Eight NeuronCore-v3" per chip [S6-15]. NeuronCore generation: NeuronCore-v3 [S6-16].
- Instance: trn2.48xlarge / trn2u.48xlarge, "16 Trainium2 chips connected using a high-bandwidth, low-latency NeuronLink-v3", 192 vCPUs, 2,048 GiB host memory [S6-18][S6-19]; trn2.3xlarge: 1 Trainium2 chip, 12 vCPUs, 128 GiB [S6-19]. Trn2 UltraServer: four trn2u.48xlarge, 64 chips, 8,192 GiB host memory [S6-18].
- Framework label: `tilebench.hardware.detect_arch()` returns `None` on a host without a CUDA/HIP device (Trainium hosts); `supports_tma()` / `supports_tmem()` are False there [S5]. The arch string `trainium2` in this skill's frontmatter is a skill label, not a `hardware.py` value.

## Execution model (native terms)

| Item | Value | Scope | Source |
|---|---|---|---|
| NeuronCores (NeuronCore-v3) | 8 | per chip | [S6-15] |
| Tensor Engine | 1; "128x128 processing elements", "effectively presenting a 256x128 systolic array to the programmer" in double-FP8 mode | per NeuronCore | [S6-17] |
| Vector Engine | 1; data-path width "512 BF16/FP16 input/output; 256 input/output for other data types" | per NeuronCore | [S6-17] |
| Scalar Engine | 1; data-path width "128 input/output" | per NeuronCore | [S6-17] |
| GpSimd Engine | "Eight fully-programmable 512-bit wide vector processors" | per NeuronCore | [S6-16][S6-17] |
| DMA engines | 128 per chip; "each NeuronCore-v3 is typically paired with 16x DMA engines" | per chip / per NeuronCore | [S6-17] |
| Collective-communication cores (CC-Cores) | "16 CC-Cores" [S6-15]; "20 CC-Cores" [S6-17] (vendor pages disagree; both quoted) | per chip | [S6-15][S6-17] |
| NeuronLink-v3 | 4 links; "1.28 TB/sec bandwidth per chip" [S6-15]; "1,024 GB/sec per chip" intra-instance, "256 GB/sec per chip" inter-instance (UltraServer) [S6-18] | per chip | [S6-15][S6-17][S6-18] |
| Logical NeuronCore (LNC) configuration | supported: "combine the compute and memory resources of multiple physical NeuronCores into a single logical NeuronCore"; available LNC values: unknown | per instance | [S6-15] |
| Engine clock | unknown | — | — |

## Compute capabilities by dtype (capability only; DSL support to be confirmed per Reference Skill)

- Tensor Engine dtypes: "cFP8, FP16, BF16, TF32, FP32" [S6-16]; NKI guide: "FP8_E4, FP8_E5, BF16, FP16, TF32, FP32" [S6-17]. Per NeuronCore: "158 cFP8 TFLOPS, and 79 BF16/FP16/TF32 TFLOPS"; structured sparsity "up to 316 TFLOPS" [S6-16].
- Per chip (8 NeuronCores), dense: "1,299 FP8 TFLOPS", "667 BF16/FP16/TF32 TFLOPS", "181 FP32 TFLOPS"; sparse: "2,563 FP8/FP16/BF16/TF32 sparse TFLOPS" [S6-15]. Per instance (16 chips): FP8 20.8 PFLOPS; FP16/BF16/TF32 10.7 PFLOPS; sparse 41 PFLOPS; FP32 2.9 PFLOPS [S6-18].
- Vector Engine dtypes: "cFP8, FP16, BF16, TF32, FP32, INT8, INT16, INT32"; "1 TFLOPS of FP32 computations" per NeuronCore [S6-16].
- Scalar Engine dtypes: same list as Vector Engine; "1.2 TFLOPS of FP32 computations" per NeuronCore [S6-16].
- INT8 on the Tensor Engine: not stated; unknown. FP64: not stated on any fetched page; unknown. MXFP4/6/8 microscaling: not stated for NeuronCore-v3; unknown.

## Memory hierarchy

| Level | Value | Scope | Source |
|---|---|---|---|
| SBUF (state buffer, on-chip SRAM) | "28MiB (or, 128 partitions of 224KiB)" [S6-17]; "28MB of on-chip SRAM" [S6-16] | per NeuronCore | [S6-16][S6-17] |
| SBUF partitions | 128 partitions x 224 KiB | per NeuronCore | [S6-17] |
| SBUF total | "224" MiB (= 8 x 28 MiB) | per chip | [S6-15] |
| PSUM (matmul accumulation buffer) | "2MiB" | per NeuronCore | [S6-17] |
| PSUM banks / bytes per bank | unknown | per NeuronCore | — |
| Registers / caches | no register file or cache levels stated by the vendor pages; unknown | — | — |
| HBM capacity | "96 GiB of device memory" per chip [S6-15]; "4 HBM stacks ... 96GiB" [S6-17]; "1,536" GiB per 16-chip instance [S6-18] | per chip / per instance | [S6-15][S6-17][S6-18] |
| HBM peak bandwidth | "2.9 TB/sec" per chip [S6-15]; "46.4" TB/sec per 16-chip instance (= 2.9 per chip) [S6-18]; NKI guide rounds to "3TB/s" [S6-17] | per chip / per instance | [S6-15][S6-17][S6-18] |
| DMA bandwidth | "3.5 TB/sec of DMA bandwidth" | per chip | [S6-15] |

Discrepancy flagged, not resolved: the EC2 instance-types page lists trn2.48xlarge "Accelerator memory" as "8192 GiB (16 x 512 GiB)" [S6-19], which does not match the Neuron documentation's 96 GiB per chip [S6-15][S6-17][S6-18]. The Neuron figures are used in this skill.

## Data movement

- HBM <-> SBUF traffic is carried by DMA engines (128 per chip, 16 paired with each NeuronCore), which "move data within and across devices" [S6-17]. Aggregate DMA bandwidth 3.5 TB/s per chip [S6-15].
- Tensor Engine reads SBUF operands and accumulates into PSUM (2 MiB per NeuronCore) [S6-17].
- Device-to-device: 4x NeuronLink-v3 per chip [S6-17]; collective communication orchestrated by CC-Cores [S6-15][S6-17].
- Which of these paths NKI exposes (e.g. `nisa.dma_copy`, direct SBUF/PSUM allocation): not asserted here; see the NKI Reference Skill.

## Limits

- SBUF: 128 partitions, 224 KiB per partition (28 MiB) per NeuronCore [S6-17]. PSUM: 2 MiB per NeuronCore [S6-17]. Tensor Engine array: 128x128 PEs (256x128 effective in double-FP8 mode) [S6-17]. Vector Engine width 512 (BF16/FP16) / 256 (other) elements; Scalar Engine width 128 [S6-17].
- Instruction-issue model, maximum tensor sizes per instruction, engine clocks: unknown.

## Software snapshot

No runtime capture exists for this snapshot: every field is **unknown** (Neuron SDK release, `neuronx-cc` compiler version, `aws-neuronx-runtime-lib`, `aws-neuronx-dkms` driver, `aws-neuronx-tools` / `neuron-profile`, `torch-neuronx`, `torch`, `neuronx-distributed`, `nki` package version, Python version, OS / kernel, instance type and LNC setting). A capture on a Trn2 host must record all of these; the exact commands are in `skills/device/CAPTURE_CHECKLIST.md`.

## Sources

1. **[S5]** `tilebench/hardware.py` @ ea04fb36 (worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt`): `device_info()` / `detect_arch()` return `None` without a CUDA/HIP device.
2. **[S6-15]** AWS Neuron documentation, "Trainium2 Architecture" — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium2.html (accessed 2026-10-05).
3. **[S6-16]** AWS Neuron documentation, "NeuronCore-v3 Architecture" — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v3.html (accessed 2026-10-05).
4. **[S6-17]** AWS Neuron documentation, NKI "Trainium2 Architecture Guide" — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium2_arch.html (accessed 2026-10-05).
5. **[S6-18]** AWS Neuron documentation, "Trn2 Architecture" (instances and UltraServer) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn2-arch.html (accessed 2026-10-05).
6. **[S6-19]** Amazon EC2 "Specifications for accelerated computing instances", Trn2 rows — https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html (accessed 2026-10-05).
7. Failed fetch (2026-10-05): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html (HTTP 404; superseded by [S6-16]).
