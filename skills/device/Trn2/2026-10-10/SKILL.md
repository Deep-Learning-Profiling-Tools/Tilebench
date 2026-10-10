---
name: trn2-device-context
device: Trn2
snapshot: "2026-10-10"
kind: device_context
arch: trainium2
revised: 2026-10-10
supersedes: "2026-10-05"
---

# Trn2 (AWS Trainium2) Device Context (snapshot 2026-10-10)

Sourced hardware and environment facts only, in AWS Neuron terms (chip, physical and logical NeuronCore,
Tensor / Vector / Scalar / GpSimd engines, SBUF, PSUM, HBM, DMA). Every number carries a unit, a scope and a
source tag `[S<n>]` (see Sources). `unknown` means no allowed source states the value. How NKI exposes a
mechanism is described in the NKI Reference Skill, not here. No benchmark-derived guidance, no measured
throughput of this benchmark, no scoring targets. `verified_on_device: true` (runtime capture [S1]).

## Identity and the resources one candidate uses

- Instance: `trn2.3xlarge` with ONE Trainium2 chip (Neuron device 0), 12 vCPUs, 128 GiB host memory [S1][S13].
  Device memory 96 GiB [S1][S2].
- Chip: 8 physical NeuronCore-v3 [S1][S2].
- Logical NeuronCore (LNC) configuration: the supported values are 1 and 2; 2 is the Trn2 default [S4].
  This host runs LNC=2: the runtime exposes 4 logical NeuronCores (IDs 0-3), each made of 2 physical
  NeuronCore-v3 [S1][S4].
- **Evaluation scope**: each candidate runs in its own process that owns exactly ONE logical NeuronCore
  (`NEURON_RT_NUM_CORES=1`, `NEURON_LOGICAL_NC_CONFIG=2`, compiler flags `--target trn2 --lnc 2`) [S1][S11].
  That is 2 physical NeuronCore-v3 plus the one 24 GB HBM bank they share [S4]. The other 3 logical cores are
  not available to the candidate. Neuron Explorer profiles of NKI kernels launched this way on this host show
  instruction streams executing on both physical cores of the logical core [S8].

| Resource | per physical NeuronCore-v3 | per logical NeuronCore (LNC=2) = one candidate | per chip |
|---|---|---|---|
| Tensor Engine (128x128 processing elements) | 1 [S3] | 2 (one per physical core) | 8 |
| Vector Engine | 1 [S3] | 2 | 8 |
| Scalar Engine | 1 [S3] | 2 | 8 |
| GpSimd Engine (8 programmable 512-bit vector processors) | 1 [S3][S6] | 2 | 8 |
| SBUF (on-chip SRAM) | 28 MiB = 128 partitions x 224 KiB [S3]; allocatable by NKI: 212,984 B per partition (`nl.tile_size.sbuf_fmax_bytes`) [S7] | 2 x 28 MiB, each attached to its own physical core (not one pooled buffer) [S3][S6] | 224 MiB [S2] |
| PSUM (matmul accumulation SRAM) | 2 MiB = 128 partitions x 16 KiB; 8 banks of 2 KiB per partition (one bank = 512 fp32 per partition) [S3][S7][S10] | 2 x 2 MiB, each attached to its own physical core | 16 MiB |
| DMA engines | 16 [S3] | 32 | 128 [S3] |
| HBM | (shared, see right) | 1 bank of 24 GB, shared by the logical core's 2 physical cores (one address space under LNC=2) [S4] | 4 banks, 96 GiB; 2.9 TB/s aggregate [S2][S4] |

## Compute engines (per physical NeuronCore-v3)

| Engine | Data-path width (elements/cycle) [S3] | Clock [S3] | Role |
|---|---|---|---|
| Tensor (TensorE) | input 2x128 (dense BF16/FP16), 4x128 (dense FP8_E4/FP8_E5), 5x128 (sparse); output 1x128 | 2.4 GHz | matrix multiplication and transposition; reads SBUF, writes PSUM [S3][S10] |
| Vector (VectorE) | 512 for BF16/FP16 input/output; 256 for other data types | 0.96 GHz | elementwise, reductions, data reshaping across 128 lanes (one per partition) [S3][S10] |
| Scalar (ScalarE) | 128 | 1.2 GHz | elementwise activation functions (with scale/bias) [S3][S10] |
| GpSimd | 8 x 512-bit programmable vector processors | 1.2 GHz | general-purpose / custom SIMD code [S3][S6] |

- Vendor throughput per NeuronCore-v3, TensorE: 158 FP8, 79 BF16/FP16/TF32 and 20 FP32 dense TFLOPS, 316 sparse
  TFLOPS [S3]. VectorE "1 TFLOPS of FP32 computations", ScalarE "1.2 TFLOPS of FP32 computations" per NeuronCore
  [S6] (the counting convention is not stated).
- Vendor throughput per chip, dense: 1,299 FP8, 667 BF16/FP16/TF32 and 181 FP32 TFLOPS; 2,563 sparse TFLOPS
  [S2]. Vendor inconsistency, not resolved: the per-chip figures are larger than 8 x the per-core TensorE
  figures, and AWS does not state the composition. Both are quoted.
- TensorE FP32 throughput is one quarter of its BF16/FP16 throughput (20 vs 79 TFLOPS per core) [S3].
- FP8 double-throughput mode: contraction depth 256 per matmul instruction (a 256x128 array presented to the
  programmer), FP8 inputs only, FP32 accumulation [S3][S5].
- Engines run their instruction streams concurrently; synchronization between engines and DMA is handled
  by the compiler/runtime [S10]. Both VectorE and ScalarE can access PSUM at full bandwidth in parallel unless
  their accesses collide on the same PSUM bank [S3].

## Data types

- TensorE input dtypes on NeuronCore-v3: `float8_e4m3` (Neuron FP8_E4), `float8_e5m2`, `bfloat16`, `float16`,
  `tfloat32` and `float32` [S5]. FP8_E3 is supported at BF16/FP16 speed [S3]. The matmul output (PSUM
  destination) must be `float32` on NeuronCore-v3 [S5].
- NOT supported by the NeuronCore-v3 TensorE:
  - integer (`int8`/`uint8`) matmul inputs: not in the documented input list [S5];
  - OCP `float8_e4m3fn`: documented only from NeuronCore-v4 on; legacy `float8_e4m3` and `float8_e4m3fn` cannot
    be mixed [S5].
- Vector and Scalar engines do all arithmetic in FP32: inputs are cast to FP32 and results cast to the
  destination dtype at no extra cost [S9][S10]. GpSimd processors can also compute natively on narrower
  integer/float types (per 512-bit processor: 16 x FP32/INT32, 32 x FP16/INT16, 64 x INT8 lanes) [S10].
  Trn2 adds native execution of 64-bit integers (INT64/UINT64) [S12].
- Native PyTorch on this stack (`torch.device("neuron")`): eager tensor operations dispatch to the device for
  fp32, fp16, bf16, int32 and bool. Copying, making contiguous or converting `int8` or FP8 tensors with
  PyTorch (`copy_`, `.contiguous()`, `.to()`) can round-trip through the host CPU, and some aten ops have no
  device implementation (for example `argmax`, `max_pool2d_with_indices`, stable `sort`) and run on the host [S9].

## Memory hierarchy and data movement

- HBM <-> SBUF traffic is carried by the DMA engines (16 per physical core). Compute engines read and write
  SBUF; TensorE writes its results to PSUM; VectorE / ScalarE read PSUM and write SBUF [S3][S10].
- SBUF and PSUM are two-dimensional: 128 partitions x a free dimension. The partition dimension of an
  on-chip tile is at most 128; each Vector/Scalar lane processes one partition [S10].
- HBM bandwidth:
  - AWS documentation publishes only the chip aggregate: 2.9 TB/s over 4 HBM banks [S2]. The NKI guide rounds
    it to "3 TB/s" [S3].
  - A per-bank figure is not in the documentation. The AWS Neuron Explorer hardware model embedded in this
    host's profile metadata uses 716 GB/s per HBM stack, and 435 GB/s of DMA bandwidth per physical
    NeuronCore [S8]. These are tool constants, not measurements.
  - DMA aggregate: 3.5 TB/s per chip [S2].
- Per-instruction size limits [S7][S10]:
  - TensorE: the stationary operand's free size is at most 128; the moving operand's free size is at most 512;
    the contraction (partition) size is at most 128, or 256 in FP8 double mode.
  - PSUM: one matmul result tile per partition fits one PSUM bank (512 fp32 = 2 KiB); a PSUM tile's free size
    is at most 4,096 elements.
- Inter-device links (NeuronLink-v3) and collective-communication cores exist per chip [S2] but play no role
  in a single-logical-core task.

## Execution and measurement on this host

- NKI kernels, and any PyTorch eager operation called inside `run()`, are compiled to Neuron executables
  (NEFFs) and executed by the Neuron runtime on the logical NeuronCore. The first call with a new shape
  compiles; it is outside the timed calls [S11].
- Measured quantity (evaluator, Trn2):
  - per `run()` call, the sum of the device durations of every Neuron execution that call issues. An execution
    that runs on both physical cores is counted once (its per-core intervals are merged);
  - one warmup call, then the mean of three timed calls;
  - host-side time is not counted. No cache flush is applied: inputs reside in HBM, and SBUF/PSUM are managed
    by the program [S11].
- The evaluator tracks PyTorch dispatch inside every `run()` call. A tensor operation that falls back to the
  host CPU, or a device-to-host copy, is detected and treated as torch computation outside the kernel (system
  rule 4) [S11].

## Software snapshot (captured 2026-10-10) [S1]

- Python environment `/home/ubuntu/workspace/native_venv` (Python 3.12.3). Pre-release AWS Neuron stack ("Beta
  5"); public documentation corresponds to Neuron 2.32.0 / NKI 0.6.0:
  - nki `0.7.0b1+34883233359.g2aa6a82a`;
  - neuronx-cc `2.0.404830.0a0+55f4bf89`;
  - torch `2.14.0+cpu` with torch-neuronx `2.14.3.0.0+unknown.dev` (native PyTorch on `torch.device("neuron")`;
    no torch-xla, torch.compile not used).
- Neuron system packages:
  - aws-neuronx-runtime-lib `2.x.79381.0-fbc4aac0b`;
  - aws-neuronx-collectives `2.x.78358.0-d99c038af`;
  - aws-neuronx-dkms (driver) `2.x.11820.0`;
  - aws-neuronx-tools `2.32.28.0-526c2b7f6`.
- OS: Ubuntu 24.04.4 LTS, kernel 6.17.0-1019-aws.
- Runtime settings of a candidate process: `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_RT_NUM_CORES=1`,
  `NEURON_CC_FLAGS="--target trn2 --lnc 2"`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` [S1][S11].

## Sources

1. **[S1]** Runtime capture on this host, 2026-10-10 (`skills/device/Trn2/2026-10-10/runtime_capture.json`):
   neuron-ls (instance type, LNC 2, 4 logical cores, 96 GB), sysfs core_count 8, pip / dpkg package versions,
   `neuronx-cc --version`, kernel, host CPU/memory; instance id omitted.
2. **[S2]** AWS Neuron documentation, "Trainium2 Architecture" — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html (accessed 2026-10-10).
3. **[S3]** AWS Neuron documentation, NKI "Trainium2 Architecture Guide" (Table 10: engine widths and clocks; SBUF 28 MiB; PSUM 2 MiB; TensorE TFLOPS; double FP8; PSUM bank access) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html (accessed 2026-10-10).
4. **[S4]** AWS Neuron documentation, "Logical NeuronCore configuration" (LNC 1 or 2, default 2; four 24 GB HBM banks, each shared by two physical NeuronCore-v3; one address space under LNC=2) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html (accessed 2026-10-10).
5. **[S5]** AWS Neuron documentation, `nki.isa.nc_matmul` (input dtypes per NeuronCore generation, FP32 PSUM destination on v2/v3, double_row FP8 mode) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html (accessed 2026-10-10).
6. **[S6]** AWS Neuron documentation, "NeuronCore-v3 Architecture" (engines, GpSimd 8 x 512-bit, engine dtype lists and TFLOPS) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html (accessed 2026-10-10).
7. **[S7]** Installed nki 0.7.0b1, `nki/language/_tile_size.py` (gen3: SBUF 128 x 224 KiB, allocatable 212,984 B per partition; PSUM 8 banks x 2 KiB per partition; max PSUM free size 4,096; gemm stationary/moving free 128/512; pmax 128).
8. **[S8]** Neuron Explorer profiles of 71 NKI cases on this host (2026-10-09). Profile metadata hardware model: HBM 716 GB/s per stack, DMA 435 GB/s and 16 DMA engines per physical core, engine clocks as in [S3]; two physical cores executing per logical core.
9. **[S9]** AWS Neuron documentation, `nki.isa.tensor_scalar` / `nki.isa.activation` (FP32 internal arithmetic on Vector/Scalar engines), and the native PyTorch fallback classification on this stack, 2026-10-06 (`torch_neuronx.get_fallback_ops()`, executed host copies).
10. **[S10]** AWS Neuron documentation, NKI "Trainium/Inferentia2 Architecture Guide" (SBUF/PSUM as 128-partition 2-D memories, 16 KiB PSUM per partition, 2 KiB PSUM bank = 512 fp32, engine roles, Vector lanes) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html (accessed 2026-10-10).
11. **[S11]** TileArena Trn2 evaluator (`tilebench/llm/v2/evaluation/neuron.py`, `neuron_worker.py`; `tilebench/core/neuron_native.py`, `neuron_intervals.py`): launch settings, busy-sum timing definition, host-fallback tracking.
12. **[S12]** AWS Neuron documentation, "Data types" (Trn2 adds native INT64/UINT64) — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html (accessed 2026-10-10).
13. **[S13]** Amazon EC2 accelerated computing instance specifications, Trn2 rows (trn2.3xlarge: 1 chip, 12 vCPUs, 128 GiB) — https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html (accessed 2026-10-05).
