---
name: b200-device-context
device: B200
snapshot: "2026-10-06"
kind: device_context
arch: blackwell
revised: 2026-10-06
---

# B200 Device Context (snapshot 2026-10-06)

Sourced hardware and environment facts only. Every number carries a unit, a scope and a
source tag `[S<n>]` (see Sources). `unknown` means no allowed source states the value.
Whether a DSL (Triton, cuTile, TileLang) exposes a mechanism listed here is NOT asserted in
this file; confirm it in the backend Reference Skill. No benchmark-derived guidance, no
measured throughput of this benchmark, no scoring targets.

## Identity

- Product name: "NVIDIA B200" (`torch.cuda.get_device_properties().name`) [S1].
- Architecture: NVIDIA Blackwell, compute capability 10.0 [S1]. Framework label: `tilebench.hardware.detect_arch()` maps capability (10, 0) to `"blackwell"` [S5].
- Capture host: dgx003, 2026-10-05, runtime capture (torch + nvidia-smi) [S1]. `verified_on_device: true`.
- Die: "Two reticle-limited dies connected by a 10 terabytes per second (TB/s) chip-to-chip interconnect", "208 billion transistors" [S6-6]. Whether and how the L2 / SM set is partitioned across the two dies: unknown.

## Execution model (native terms)

| Item | Value | Scope | Source |
|---|---|---|---|
| Streaming Multiprocessors (SMs) | 148 | whole device | [S1] |
| Warp size | 32 threads | per warp | [S1][S6-1] |
| Max resident threads | 2048 | per SM | [S1][S6-1] |
| Max resident warps | 64 | per SM | [S6-1][S6-2] |
| Max resident thread blocks | 32 | per SM | [S6-1][S6-2] |
| Max threads per thread block | 1024 | per block | [S1][S6-1] |
| Max block dimensions (x, y, z) | 1024, 1024, 64 | per block | [S6-1] |
| Max grid dimensions (x; y or z) | 2^31-1; 65535 | per grid | [S6-1] |
| Thread block cluster, portable max | 8 blocks | per cluster | [S6-2] |
| Max SM clock | 1965 MHz | whole device | [S1] |
| Memory clock | 3996 MHz | whole device | [S1] |
| Memory bus width | 7680 bit | whole device | [S1] |

## Compute capabilities by dtype (capability only; DSL support to be confirmed per Reference Skill)

- Matrix unit: 5th-generation Tensor Core, PTX `tcgen05` family, "sm_100a / sm_100f" [S6-3]. `tcgen05.mma` kinds: `kind::f16`, `kind::tf32`, `kind::f8f6f4`, `kind::i8`, `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4`, `kind::ti16` [S6-3]. The `.f8f6f4type` operand set is `{.e4m3, .e5m2, .e3m2, .e2m3, .e2m1}`; block-scaled kinds use `.e2m1` operands with `.ue8m0` scales (`mxf4`) or `.ue8m0`/`.ue4m3` scales (`mxf4nvf4`) [S6-3]. The verbatim operand table of `kind::f16` was not captured in the fetch; FP16/BF16 Tensor Core throughput is listed by [S6-4].
- Tensor-Core dtypes with vendor throughput figures: FP4, FP8, FP6, INT8, FP16, BF16, TF32, FP64 [S6-4]. "Second-generation Transformer Engine", FP4 and "community-defined microscaling formats" [S6-6].
- Vendor peak throughput, HGX B200 platform page, 8-GPU totals, footnote "1. Sparse specification. 2. Dense specification." [S6-4]. The page shows FP4 as sparse | dense explicitly; the FP8/FP6, INT8, FP16/BF16 and TF32 rows carry the sparse footnote, so their dense value is one-half of the displayed figure. Per-GPU values are the 8-GPU dense totals divided by 8 (derived, not stated by the page):

| dtype | HGX B200 displayed (8 GPUs) | 8-GPU dense | per-GPU dense (derived) |
|---|---|---|---|
| FP4 Tensor Core | 144 \| 72 PFLOPS (sparse \| dense) | 72 PFLOPS | 9 PFLOP/s |
| FP8 / FP6 Tensor Core | 72 PFLOPS (sparse) | 36 PFLOPS | 4.5 PFLOP/s |
| INT8 Tensor Core | 72 POPS (sparse) | 36 POPS | 4.5 POP/s |
| FP16 / BF16 Tensor Core | 36 PFLOPS (sparse) | 18 PFLOPS | 2.25 PFLOP/s |
| TF32 Tensor Core | 18 PFLOPS (sparse) | 9 PFLOPS | 1.125 PFLOP/s |
| FP32 | 600 TFLOPS | 600 TFLOPS | 75 TFLOP/s |
| FP64 / FP64 Tensor Core | 296 TFLOPS | 296 TFLOPS | 37 TFLOP/s |

  These are vendor specifications of the hardware (dense, no 2:4 sparsity), not measured or achievable rates of any kernel, and not the scoring targets of this benchmark.

## Memory hierarchy

| Level | Value | Scope | Source |
|---|---|---|---|
| 32-bit registers | 65,536 (64 K; 256 KiB) | per SM | [S1][S6-1][S6-2] |
| Max 32-bit registers | 255 | per thread | [S6-1][S6-2] |
| Max 32-bit registers | 64 K | per thread block | [S6-1] |
| Shared memory, default limit | 49,152 B (48 KiB) | per block | [S1] |
| Shared memory, opt-in max | 232,448 B (227 KiB); "227 KB" | per block | [S1][S6-1][S6-2] |
| Shared memory | 233,472 B (228 KiB); "228 KB" | per SM | [S1][S6-1][S6-2] |
| Shared memory banks | 32 | per SM | [S6-1] |
| Unified L1 / texture / shared cache | 256 KB | per SM | [S6-2] |
| Tensor Memory (TMEM) | 512 columns x 128 lanes, 32-bit cells = 256 KiB (derived) | per CTA-visible TMEM of one SM ("per CTA" in PTX wording) | [S6-3] |
| Local memory max | 512 KB | per thread | [S6-1] |
| Constant memory | 64 KB; cache working set 8 KB per SM | whole device / per SM | [S6-1] |
| L2 cache | 132,644,864 B (126.5 MiB) | whole device (runtime-reported) | [S1] |
| HBM3e capacity | 191,495,471,104 B (torch total_memory, 178.3 GiB); 183,359 MiB (nvidia-smi); 1,440 GB total / 8 GPUs = 180 GB (derived) | whole device | [S1][S6-5] |
| HBM3e bandwidth, vendor specification | DGX B200 "64 TB/s HBM3e bandwidth" total / 8 GPUs = 8 TB/s per GPU (derived) | whole device | [S6-5] |

Framework facts [S5]: `last_level_cache_bytes()` returns the runtime `L2_cache_size` on Blackwell (no `_LLC_BYTES` override), i.e. 132,644,864 B; the timer's eviction buffer is "2x" that size (rule stated in the `hardware.py` comment).

## Data movement

- Tensor Memory Accelerator (TMA): PTX `cp.async.bulk.tensor` "Requires sm_90 or higher"; the `.multicast::cluster` qualifier is "advised to be used with .target sm_90a or sm_100f or sm_100a ..." [S6-3]. Framework: `supports_tma()` is True for `blackwell` [S5].
- Tensor Memory load/store and allocation: TMEM "must be allocated by a single warp in a CTA"; "The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2. When a column is allocated, all 128 lanes of the column are allocated."; all allocated TMEM "must be explicitly deallocated before the kernel exits" [S6-3]. Framework: `supports_tmem()` is True only for `blackwell` [S5].
- Thread block clusters (portable size 8) and distributed shared memory (a block may access the shared memory of other blocks in its cluster) [S6-2].
- Hardware decompression engine ("LZ4, Snappy, and Deflate") [S6-6] — not a kernel-level mechanism.
- Asynchronous non-tensor bulk copies (`cp.async`, `cp.async.bulk`): not fetched from vendor docs for this snapshot; capability: unknown.

## Limits

- 1024 threads per block; 2048 threads, 64 warps, 32 blocks per SM; 255 registers per thread; 64 K registers per block and per SM; 227 KB opt-in shared memory per block; 228 KB shared memory per SM [S1][S6-1][S6-2].
- Max SM clock 1965 MHz; memory clock 3996 MHz [S1]. Base/boost clock schedule under load: unknown.
- TMEM: 512 columns per SM, allocation unit 32 columns, power-of-two column counts [S6-3].

## Software snapshot (host dgx003, captured 2026-10-05) [S1]

| Component | Version |
|---|---|
| NVIDIA driver | 595.58.03 |
| CUDA runtime (torch.version.cuda) | 13.0 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| cuda-tile | 1.5.0 |
| tilelang | 0.1.11 |
| Python | unknown |
| apache-tvm-ffi | unknown |
| tileiras (CUDA Tile compiler used by cuda-tile) | 13.2 [S7] |
| nvcc / nvvm | unknown |
| OS / kernel | unknown |
| Proton backend | unknown |

APIs whose documented minimum tileiras version is 13.3 or later are unavailable in this B200 environment.

## Sources

1. **[S1]** Runtime capture on dgx003, 2026-10-05: `torch.cuda.get_device_properties(0)` + `nvidia-smi` (name, compute capability, multi_processor_count, warp_size, max_threads_per_block, max_threads_per_multi_processor, regs_per_multiprocessor, shared_memory_per_block, shared_memory_per_block_optin, shared_memory_per_multiprocessor, L2_cache_size, total_memory, clock_rate, memory_clock_rate, memory_bus_width, driver, torch/CUDA/triton/cuda-tile/tilelang versions).
2. **[S5]** `tilebench/hardware.py`, worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt` at HEAD ea04fb36 (`_NVIDIA_ARCH`, `supports_tma`, `supports_tmem`, `last_level_cache_bytes`).
3. **[S6-1]** NVIDIA CUDA C Programming Guide v12.9.0 (archived single page), "Technical Specifications per Compute Capability", column 10.x — https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#features-and-technical-specifications (accessed 2026-10-05). The current v13.4.2 split page `05-appendices/compute-capabilities.html` returned HTTP 404 on 2026-10-05.
4. **[S6-2]** NVIDIA Blackwell Tuning Guide — https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html (accessed 2026-10-05).
5. **[S6-3]** NVIDIA PTX ISA, sections 9.7.18 (TensorCore 5th Generation: Tensor Memory, `tcgen05.mma` kinds, allocation) and `cp.async.bulk.tensor` Target ISA Notes — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (accessed 2026-10-05).
6. **[S6-4]** NVIDIA HGX platform page, HGX B200 column and footnotes ("1. Sparse specification. 2. Dense specification.") — https://www.nvidia.com/en-us/data-center/hgx/ (accessed 2026-10-05).
7. **[S6-5]** NVIDIA DGX B200 page ("1,440 GB total, 64 TB/s HBM3e bandwidth", 8 GPUs) — https://www.nvidia.com/en-us/data-center/dgx-b200/ (accessed 2026-10-05).
8. **[S6-6]** NVIDIA Blackwell architecture page — https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ (accessed 2026-10-05).
9. **[S7]** `tileiras --version` on dgx003, 2026-10-07: "Cuda compilation tools, release 13.2, V13.2.78" (the compiler cuda-tile uses on this host).
