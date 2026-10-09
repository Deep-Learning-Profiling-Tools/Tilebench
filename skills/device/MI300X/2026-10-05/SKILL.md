---
name: mi300x-device-context
device: MI300X
snapshot: "2026-10-05"
kind: device_context
arch: cdna3
revised: 2026-10-09
---

# MI300X Device Context (snapshot 2026-10-05)

Sourced hardware and environment facts only, in AMD native terms (XCD, CU, SIMD, wavefront,
LDS, VGPR/AGPR/SGPR, Infinity Cache). Every number carries a unit, a scope and a source tag
`[S<n>]` (see Sources). `unknown` means no allowed source states the value. Whether a DSL
(Triton, TileLang) exposes a mechanism listed here is NOT asserted in this file; confirm it in
the backend Reference Skill. No benchmark-derived guidance. Revised 2026-10-09 on the campaign device
itself (host "7"): limits, LDS per work-group and instruction support marked [S7] come from that
device's own runtime output and executed code: `verified_on_device: true`.

## Identity

- Product: AMD Instinct MI300X; rocminfo marketing name "AMD Instinct MI300X VF" ("VF" = virtual function, i.e. the capture ran on an SR-IOV-exposed device); torch device name "AMD Radeon Graphics" [S4].
- ISA / target: `gfx942:sramecc+:xnack-` (gcnArchName); architecture CDNA3; LLVM target `gfx942`; GFXIP 9.4 [S4][S6-11]. Framework label: `tilebench.hardware.detect_arch()` maps `gfx942` to `"cdna3"` [S5]; capture records `detect_arch: "cdna3"` [S4].
- Capture host: hostname "7", Ubuntu 24.04.4 LTS, kernel 6.8.0-138-generic, captured 2026-10-01 [S4].

## Execution model (native terms)

| Item | Value | Scope | Source |
|---|---|---|---|
| Accelerator Complex Dies (XCDs) | 8 (KFD `num_xcc` 8) | whole device | [S4][S6-12] |
| Compute Units (CUs) | 304 active ("40 / 38" physical / active per XCD) | whole device / per XCD | [S4][S6-11][S6-12] |
| SIMDs | 4 ("grouped into 2 SIMD pairs") | per CU | [S6-12] |
| Matrix Cores | 1,216 (= 4 per CU, derived) | whole device | [S6-12] |
| Wavefront size | 64 work-items | per wavefront | [S4][S6-11][S6-14] |
| Max work-group size | 1024 work-items | per work-group | [S6-14] |
| Max engine clock | 2,100 MHz | whole device | [S6-12] |
| Max resident wavefronts | 8 per SIMD; 32 per CU (KFD `max_waves_per_simd` 8; rocminfo "Max Waves Per CU: 32") | per SIMD / per CU | [S7] |
| Max resident work-items | 2,048 (rocminfo "Max Work-item Per CU"; torch `max_threads_per_multi_processor` 2048) | per CU | [S7] |
| Max work-groups per CU | unknown | — | — |

Occupancy rule stated by the vendor: "each Execution Unit (EU) has 512 available VGPRs, which are allocated in blocks of 16" (EU = SIMD); example: "the occupancy is limited to 2 waves per EU because (176 x 3 > 512)" [S6-12].

## Compute capabilities by dtype (capability only; DSL support to be confirmed per Reference Skill)

- Matrix Cores (MFMA) dtypes: FP64, FP32, TF32 (XF32), FP16, BF16, FP8, INT8 [S6-12][S6-13]; HIP feature table for CDNA3: Matrix Cores yes, Float16 yes, BFloat16 yes, 8-bit floating point yes, Tensor float32 yes [S6-14].
- Vendor peak table, "FLOPS/CLOCK/CU" and "Peak TFLOPS" (whole device; sparsity not mentioned) [S6-13]:

| Unit / dtype | FLOPS per clock per CU | Peak TFLOPS (whole device) |
|---|---|---|
| Matrix FP64 | 256 | 163.4 |
| Vector FP64 | 128 | 81.7 |
| Matrix FP32 | 256 | 163.4 |
| Vector FP32 | 256 | 163.4 |
| Matrix TF32, i.e. XF32 (the page labels this row "Vector TF32") | 1024 | 653.7 |
| Matrix FP16 | 2048 | 1307.4 |
| Matrix BF16 | 2048 | 1307.4 |
| Matrix FP8 | 4096 | 2614.9 |
| Matrix INT8 | 4096 | 2614.9 |

- TF32-precision math on CDNA3 is the XF32 Matrix Core format (`v_mfma_*_xf32`, e.g. `v_mfma_f32_32x32x4_xf32`); there is no vector TF32 instruction [S6-12][S7].
- Packed vector FMA: `v_pk_fma_f32` (2 fp32 lanes per instruction) and `v_pk_fma_f16` (2 fp16 lanes per instruction) are supported; both were assembled and executed on this device [S7].
- FP4 / FP6 / microscaling matrix formats: not stated on the fetched pages; capability: unknown.

## Memory hierarchy

| Level | Value | Scope | Source |
|---|---|---|---|
| VGPR file | 512 KiB (table lists per-CU resources; scope not footnoted) | per CU | [S6-11] |
| VGPRs available | 512, allocated in blocks of 16 | per SIMD (EU) | [S6-12] |
| Max architected registers | 256 vector (VGPR) + 256 matrix (AGPR) 32-bit; 104 SGPR | per work-item / per wavefront (HIP table wording "per thread") | [S6-14] |
| SGPR file | 12.5 KiB | per CU | [S6-11] |
| LDS (local data share) | 64 KiB ("64 KB") | per CU | [S6-11][S6-12] |
| Max LDS allocation per work-group | 64 KiB (65,536 B; rocminfo GROUP segment "64(0x40) KB", torch `shared_memory_per_block` 65536) | per work-group | [S7] |
| L1 vector (data) cache | 32 KiB ("L1: 32(0x20) KB" rocminfo) | per CU | [S4][S6-11][S6-12][S6-13] |
| L1 scalar cache | 16 KiB | per 2 CUs | [S6-11] |
| L1 instruction cache | 64 KiB | per 2 CUs | [S6-11] |
| L2 cache | 4 MiB ("4096(0x1000) KB" rocminfo; torch `L2_cache_size` 4,194,304 B); 32 MiB total "(4 per XCD)" | per XCD | [S4][S6-11][S6-12][S6-13] |
| AMD Infinity Cache (L3, last-level cache) | 256 MiB ("L3: 262144(0x40000) KB" rocminfo; "256 MB") | whole device, shared by all 304 CUs | [S4][S5][S6-11][S6-12] |
| HBM3 capacity | "192 GB HBM3" / "192" GiB (vendor); torch total 205,822,885,888 B (191.7 GiB); amd-smi "196288 MB" | whole device | [S4][S6-11][S6-12] |
| HBM3 stacks | 8 | whole device | [S6-13] |
| HBM3 peak bandwidth | "5.3 TB/s" ("5.3 TB per second") | whole device | [S6-12][S6-13] |

Framework facts [S5][S4]: the runtime-reported L2 (4 MiB per XCD) is NOT the last-level cache; `tilebench.hardware._LLC_BYTES["cdna3"] = 256 MiB` (268,435,456 B) is the registered LLC size, `cdna3` is in `_LLC_CALIBRATION_REQUIRED`, and the capture records `last_level_cache_bytes: 268435456`, `llc_mib: 256.0`, `flush_buffer_mb: 512` ("The timer's 2x rule gives 512 MiB").

## Data movement

- No tensor-descriptor copy engine is asserted for this device: the framework helpers `supports_tma()` and `supports_tmem()` both return False for `cdna3` [S5].
- Direct global-to-LDS loads, asynchronous copies and DMA engines: the fetched vendor pages do not state these capabilities for CDNA3; capability: unknown.
- Infinity Fabric / XCD-to-memory topology, and which XCD a work-group is dispatched to: not stated on the fetched pages; unknown.

## Limits

- Work-group: 1024 work-items max; wavefront 64 [S6-14]. Registers: 256 VGPR + 256 AGPR + 104 SGPR per work-item [S6-14]; 512 VGPRs per SIMD in blocks of 16 [S6-12]. LDS 64 KiB per CU and at most 64 KiB per work-group [S6-11][S6-12][S7]. At most 8 wavefronts per SIMD, 32 wavefronts and 2,048 work-items per CU [S7]. Max engine clock 2,100 MHz [S6-12]. Memory clock: unknown.

## Software snapshot (host "7", captured 2026-10-01) [S4]

| Component | Version |
|---|---|
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0-138-generic |
| System ROCm | 7.14.0 |
| amdgpu driver | 6.19.14.31400000 |
| HSA runtime (rocminfo) | 1.21 |
| HIP (torch.version.hip) | 7.1.25424 |
| torch | 2.10.0+rocm7.1 |
| triton | 3.6.0 |
| tilelang | 0.1.11 |
| cuda-tile | not installed (`cuda_tile: null`) |
| apache-tvm-ffi | unknown |
| Python | 3.12.14 |
| Proton backend | roctracer |

Framework timing mode on this device [S4]: configs requesting `use_cuda_graph: true` ran eager (`effective_use_cuda_graph: false`, "ROCm HIP Graph timing fallback: Proton/roctracer does not reliably attribute child kernels in graph replay").

## Sources

1. **[S4]** MI300X environment capture `results/MI300X/logs/metadata/environment.json`, read with `git show origin/archive/tilebenchpp-2026-10:results/MI300X/logs/metadata/environment.json` (accessed 2026-10-05): rocminfo / amd-smi / KFD / torch / provenance values.
2. **[S5]** `tilebench/hardware.py` @ ea04fb36 (worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt`): `_AMD_ARCH`, `_LLC_BYTES`, `_LLC_CALIBRATION_REQUIRED`, `supports_tma`, `supports_tmem`, cdna3 comment.
3. **[S6-11]** ROCm "GPU hardware specifications" table, MI300X row: `MI300X | CDNA3 | gfx942 | 192 | 304 (38 per XCD) | 64 | 64 | 256 | 32 (4 per XCD) | 32 | 16 per 2 CUs | 64 per 2 CUs | 512 | 12.5 | 9 | 4` with headers `VRAM (GiB) | Compute Units | Wavefront Size | LDS (KiB) | L3 Cache (MiB) | L2 Cache (MiB) | L1 Vector Cache (KiB) | L1 Scalar Cache (KiB) | L1 Instruction Cache (KiB) | VGPR File (KiB) | SGPR File (KiB) | GFXIP Major | GFXIP Minor` — https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html (accessed 2026-10-05).
4. **[S6-12]** ROCm "MI300X workload optimization" page (8 XCDs, 40/38 CUs per XCD, 304 CUs, 4 SIMDs per CU, 64 KB LDS per CU, 32 KB L1 per CU, 4 MB L2 per XCD, 256 MB Infinity Cache, 192 GB HBM3, 5.3 TB/s, 2,100 MHz, 1,216 matrix cores, 512 VGPRs per EU in blocks of 16) — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html (accessed 2026-10-05).
5. **[S6-13]** ROCm 6.2.4 "AMD Instinct MI300 microarchitecture" page (8 HBM3 stacks, 5.3 TB/s, 4 MB L2 per XCD, 32 KB L1, FLOPS/clock/CU peak table) — https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-arch/mi300.html (accessed 2026-10-05). The `/en/latest/` spelling of this page returned HTTP 404.
6. **[S6-14]** HIP "Hardware features" table, CDNA3 column (wavefront 64, 1024 work-items per work-group — table row "Maximum threads per block", 256 vector + 256 matrix registers, 104 scalar registers, Matrix Cores, FP16/BF16/FP8/TF32) — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html (accessed 2026-10-05).
7. **[S7]** On-device capture on the campaign device (host "7", 2026-10-09, AMD Instinct MI300X VF, gfx942, ROCm 7.14, torch 2.10.0+rocm7.1, triton 3.6.0): `rocminfo` ("Max Waves Per CU: 32", "Max Work-item Per CU: 2048", "Workgroup Max Size: 1024", Pool "Segment: GROUP" "Size: 64(0x40) KB"); `/sys/class/kfd/kfd/topology/nodes/*/properties` (`max_waves_per_simd 8`, `simd_per_cu 4`, `lds_size_in_kb 64`, `num_xcc 8`, `max_engine_clk_fcompute 2100`); `torch.cuda.get_device_properties(0)` (`shared_memory_per_block` 65536, `max_threads_per_multi_processor` 2048, `multi_processor_count` 304); AMDGCN of kernels compiled and executed on the device containing `v_pk_fma_f32`, `v_pk_fma_f16` and `v_mfma_f32_32x32x4_xf32` (calibration MI300X-20261009T060617Z-eccfca2c raw files and the 2026-10-05 MI300X IR dumps).
8. Failed fetches (2026-10-05): https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html (timeout, 3 attempts); https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf (timeout); https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html (404).
