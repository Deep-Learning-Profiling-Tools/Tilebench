---
name: gh200-device-context
device: GH200
snapshot: "2026-10-05"
kind: device_context
arch: hopper
revised: 2026-10-05
---

# GH200 Device Context (snapshot 2026-10-05)

Sourced hardware and environment facts only. Every number carries a unit, a scope and a
source tag `[S<n>]` (see Sources). `unknown` means no allowed source states the value.
Whether a DSL (Triton, cuTile, TileLang) exposes a mechanism listed here is NOT asserted in
this file; confirm it in the backend Reference Skill. No benchmark-derived guidance.
The device facts below were captured on the campaign host (not on the host that wrote this
file): `verified_on_device: false`.

## Identity

- Product name: "NVIDIA GH200 480GB" (torch device name) [S3]. Vendor: nvidia; compute capability 9.0 [S3].
- Architecture: NVIDIA Hopper GPU on the Grace Hopper Superchip. Framework label: `tilebench.hardware.detect_arch()` maps capability (9, 0) to `"hopper"` [S5]; the GH200 environment capture records `detect_arch: "hopper"` [S3].
- Superchip configuration (single-chip GH200): "96GB HBM3 | 144 GB HBM3e" GPU memory variants; "Up to 480 GB LPDDR5X" CPU memory; "72 Arm Neoverse V2 Cores with 4x 128b SVE2 per core" [S6-8]. The captured device is the 96 GB HBM3 variant (nvidia-smi 97,871 MiB) [S3]. NVLink-C2C: "900 gigabytes per second (GB/s) of coherent interface" [S6-10].
- Capture host: hostname "gracehopper", aarch64 conda environment (`miniconda3-aarch64`), captured 2026-10-02 after the campaign [S3].

## Execution model (native terms)

| Item | Value | Scope | Source |
|---|---|---|---|
| Streaming Multiprocessors (SMs) | 132 | whole device | [S3] |
| Warp size | 32 threads | per warp | [S6-1] |
| Max resident threads | 2048 | per SM | [S6-1] |
| Max resident warps | 64 | per SM | [S6-1][S6-7] |
| Max resident thread blocks | 32 | per SM | [S6-1][S6-7] |
| Max threads per thread block | 1024 | per block | [S6-1] |
| Max block dimensions (x, y, z) | 1024, 1024, 64 | per block | [S6-1] |
| Max grid dimensions (x; y or z) | 2^31-1; 65535 | per grid | [S6-1] |
| Thread block cluster | portable max 8; "nonportable cluster size of 16" on H100 | per cluster | [S6-7] |
| SM clock (max) | unknown | whole device | — |
| Memory clock / bus width | unknown | whole device | — |

## Compute capabilities by dtype (capability only; DSL support to be confirmed per Reference Skill)

- Matrix unit: 4th-generation Tensor Core; PTX asynchronous warpgroup MMA `wgmma.mma_async`, "Requires sm_90a" [S6-3]. Operand types listed in the PTX shape tables: `.f16`, `.bf16`, `.tf32`, `.e4m3` / `.e5m2` (FP8), and `.u8`/`.s8` integer operand combinations [S6-3].
- Hopper Tensor Core dtypes with vendor throughput entries (H100 SXM product table): FP64 Tensor Core, TF32, BF16, FP16, FP8, INT8 [S6-9]. Those H100 SXM figures are NOT transferred here: the GH200 GPU's own peak TFLOPS per dtype are **unknown** (the GH200 datasheet could not be fetched; see Sources).
- FP32 non-tensor rate: "2x more FP32 operations per cycle per SM than devices of compute capability 8.0" [S6-7]; absolute value: unknown.
- FP4 / FP6 / microscaling MMA kinds: not part of `wgmma` [S6-3]; capability: none stated.
- Tensor Memory (TMEM): not present; framework `supports_tmem()` is False for `hopper` [S5][S3].

## Memory hierarchy

| Level | Value | Scope | Source |
|---|---|---|---|
| 32-bit registers | 64 K | per SM | [S6-1] |
| Max 32-bit registers | 255 | per thread | [S6-1][S6-7] |
| Max 32-bit registers | 64 K | per thread block | [S6-1] |
| Shared memory, default limit | unknown (runtime `shared_memory_per_block` not captured) | per block | — |
| Shared memory, opt-in max | "227 KB" ("CUDA reserves 1 KB per block") | per block | [S6-1][S6-7] |
| Shared memory | "228 KB" | per SM | [S6-1][S6-7] |
| Shared memory banks | 32 | per SM | [S6-1] |
| Unified L1 / shared / texture cache | "256 KB" | per SM | [S6-7] |
| Local memory max | 512 KB | per thread | [S6-1] |
| Constant memory | 64 KB; cache working set 8 KB per SM | whole device / per SM | [S6-1] |
| L2 cache | 62,914,560 B (60 MiB) (`torch_L2_cache_size_bytes`) | whole device (runtime-reported) | [S3] |
| HBM3 capacity | 97,871 MiB (nvidia-smi); "96GB HBM3" | whole device | [S3][S6-8] |
| HBM3 peak bandwidth | "Up to 4 TB/s" (96 GB HBM3 variant); "Up to 4.9 TB/s" (144 GB HBM3e variant) | whole device | [S6-8] |
| CPU LPDDR5X | "Up to 480 GB", "Up to 500 GB/s" | per superchip | [S6-8] |
| NVLink-C2C (CPU-GPU) | 900 GB/s | per superchip | [S6-10] |

Note: the Hopper Tuning Guide's "50 MB" L2 refers to the H100 GPU [S6-7]; the GH200 runtime reports 60 MiB [S3], and that runtime value is the one the framework uses.

Framework facts: `last_level_cache_bytes()` returns the runtime L2 on Hopper (no `_LLC_BYTES` override) [S5]; the GH200 capture records `last_level_cache_bytes: 62914560` and `l2_flush_buffer_mb: 120` [S3].

## Data movement

- Tensor Memory Accelerator (TMA): "transfer 1D and up to 5D tensors between global memory and shared memory"; supports "reduction operations such as add/min/max" [S6-7]. PTX `cp.async.bulk.tensor` "Requires sm_90 or higher" [S6-3]. Framework: `supports_tma()` is True for `hopper` (host-side TMA tensor descriptors) [S5]; capture records `supports_tma: true` [S3].
- Thread block clusters and distributed shared memory: a block "can access the shared memory of other thread blocks within its cluster" [S6-7].
- Non-tensor async copies (`cp.async`, `cp.async.bulk`): not fetched for this snapshot; capability: unknown.
- NVLink-C2C to Grace LPDDR5X (900 GB/s) is a host-memory path, not a kernel-level mechanism [S6-10].

## Limits

- 1024 threads per block; 2048 threads, 64 warps, 32 blocks per SM; 255 registers per thread; 64 K registers per block and per SM; 227 KB opt-in shared memory per block; 228 KB shared memory per SM [S6-1][S6-7].
- Cluster size: 8 portable, 16 non-portable [S6-7].
- Clocks (SM, memory): unknown.

## Software snapshot (host "gracehopper", captured 2026-10-02) [S3]

| Component | Version |
|---|---|
| NVIDIA driver | 590.44.01 |
| CUDA runtime (torch.version.cuda) | 13.0 |
| tilelang CUDA_HOME | /usr/local/cuda-13.1 |
| torch | 2.10.0+cu130 |
| triton | 3.6.0 |
| cuda-tile | 1.5.0 |
| tilelang | 0.1.11 |
| apache-tvm-ffi | 0.1.11 |
| nvidia-cuda-tileiras / nvidia-cuda-nvcc / nvidia-nvvm | 13.4.92 |
| cuda-bindings | 13.0.3 |
| numpy | 2.2.6 |
| Python | 3.10.21 |
| cuTile `tileiras` binary | `.../envs/tilebench++_env/lib/python3.10/site-packages/nvidia/cu13/bin/tileiras` |
| OS / kernel | unknown |
| Proton backend | unknown |

Framework setting at capture: `cutile_crash_isolation_timeout_sec_at_capture: 60` [S3]. Per-run provenance sidecar `mul2_default_triton-cutile-tilelang.json` (2026-10-01) records the same torch/triton/cuda_tile/tilelang versions and device name [S3].

## Sources

1. **[S3]** GH200 environment capture `results/GH200/logs/metadata/environment.json` and provenance sidecar `results/GH200/logs/provenance/mul2_default_triton-cutile-tilelang.json`, read with `git show origin/archive/tilebenchpp-2026-10:<path>` (accessed 2026-10-05).
2. **[S5]** `tilebench/hardware.py` @ ea04fb36 (worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt`).
3. **[S6-1]** NVIDIA CUDA C Programming Guide v12.9.0 (archived single page), "Technical Specifications per Compute Capability", column 9.0 — https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#features-and-technical-specifications (accessed 2026-10-05). The current v13.4.2 split page `05-appendices/compute-capabilities.html` returned HTTP 404 on 2026-10-05.
4. **[S6-3]** NVIDIA PTX ISA (`wgmma.mma_async` Target ISA Notes and operand tables; `cp.async.bulk.tensor` Target ISA Notes) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (accessed 2026-10-05).
5. **[S6-7]** NVIDIA Hopper Tuning Guide — https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html (accessed 2026-10-05).
6. **[S6-8]** NVIDIA Grace Performance Tuning Guide, GH200 specification table — https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html (accessed 2026-10-05).
7. **[S6-9]** NVIDIA H100 product page (H100 SXM Tensor Core dtype list; figures not used) — https://www.nvidia.com/en-us/data-center/h100/ (accessed 2026-10-05).
8. **[S6-10]** NVIDIA GH200 Grace Hopper Superchip product page — https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/ (accessed 2026-10-05).
9. Failed fetches (2026-10-05): GH200 datasheet at https://resources.nvidia.com/en-us-grace-cpu/grace-hopper-superchip (302 redirect to nvidia.com root); https://docs.nvidia.com/grace-performance-tuning-guide/index.html (404; superseded by [S6-8]); https://docs.nvidia.com/dgx/dgxgh200-user-guide/introduction.html (404).
