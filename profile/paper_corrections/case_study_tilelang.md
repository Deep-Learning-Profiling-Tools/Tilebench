# TileLang case study

Paragraphs are verbatim. Only TileLang figures were corrected; Triton and cuTile
numbers are PDF-sourced and left as written. TileLang latencies come from the raw
autotune logs (`results/logs/time_measurement_logs/<op>_tilelang_*.json`), not the
torch-rescaled `tilelang_ms` CSV column.

### 1d_conv, 2d_conv, 3d_conv

In fp16, TileLang-Triton instruction count is 0.48x, 0.69x, 1.82x for 1D, 2D, 3D. In FP32, they are 1.01x, 0.77x, 0.88x. TileLang requires 6.69× and 4.66× the Triton latency (maximum input case) in 1D FP16 and FP32, 3.38× and 2.79× in 2D, and 5.03× and 2.82× in 3D.

In fp16, TileLang lowers down to 5th gen tensor ops like tcgen05.mma. TileLang does not use CuTe TMEM allocator, and omits the TMEM allocation permit release, and ends up with a 1 CTA / SM profile. Autotune selects the largest tile for matmul but not the max amount of threads which would result in more warps. This is because selecting the max amount of threads results in 1.5-1.9x number of instructions and barrier stalls increase by 3.8 - 12.6x. For 1D, 2D, 3D conv, Tilelang ALU activity reaches 2.20%, 10.72%, 18.10% while tensor activity remains below 1.0%. TileLang's kernel is not bandwidth saturated, instead the 1 CTA / SM results in too few concurrent requests to be sent to HBM resulting in (1D, 2D, 3D) 1.02%, 0.30%, 0.17% of 8 TB/S peak. All convolutions choose 4 warps per CTA for TileLang.

In fp32, TileLang does not lower down to 5th gen tensor ops like tcgen05.mma, leaving the accumulator in registers. In TileLang ALU activity is (1D, 2D, 3D) 5.77%, 8.27%, 11.67% and tensor activity is 6.45%, 6.33%, 5.05%. On all convolutions, TileLang achives 255 registers/thread, which is the B200 limit. TileLang is at 8 resident warps per SM for all convolutions. While Triton and CuTile range from 12 to 20 resident warps. For 1D conv, TileLang is at 2.5% of 8 TB/s while Triton is at 11.01% and CuTile is at 9.56%. For 2d and 3d Conv (TL, TR, Cu) it is 1.35%, 2.49%, 1.72% and 0.85%, 2.19%, 1.21%. In conclusion, the accumulator becomes on registers which reduces the number of resident warps per SM, and lowers the concurrent requests to HBM.

### radix_sort

TileLang (in maximum input case) requires 1.67 Triton latency while CuTile requires 1.90 Triton latency. Every DSL lands on 128 threads per CTA for scatter. Total NCU profiling is (TL, TR, Cu) is 135.65us, 81.47us, 154.98us while scatter (TL, TR, Cu) is 101.41 us, 50.37 us,115.46 us. Standardized to Triton, that is 1.67x, 1.00x, 1.90x for the pass and 2.01x, 1.00x, 2.29x for scatter. Scatter instruction count (TL, TR, Cu) is 46.54M, 31.88M, 73.52M, or 1.46x and 2.31x Triton. Register (TL, TR, Cu) per thread is 48, 64, 108, and TileLang has the greatest warp active NCU profile of 58.52% with Triton at 44.98% and CuTile at 23.03%. TileLang is also at 9.2 kb dyn SMEM like Triton. However, TileLang has the lowest % of Issues Active at 41.91%, Triton at 60.77% and CuTile at 58.09% possibly because of TL's 13.08 barrier stalls per issue, while Triton is at 0.79 and CuTile is at 0.38. TileLang has a much greater barrier stall per issue due to their cumsum lowering, where 3 out of 4 warps have to wait for 1 warp to finish a serial scan of the array.

### matmul_fp32_fp16_fp8

CuTile tensor pipeline ranges from 79% to 81% and Triton is 52% to 69% for FP32, FP16 and FP8. CuTile is 1.55×, 1.12×, and 1.22× faster than Triton (maximum input case) in FP32, FP16, and FP8. TileLang-Triton speedup is 0.40x, 0.80x, 0.72x in fp32, fp16 and fp8 respectively. TileLang has 68.23% tensor activity in fp32, 46.20% in fp16, 43.78% in fp8. In all dtypes, TileLang loads 0 TMA bytes. SMEM allocation (TL, TR, Cu) is (131.07, 98.35, 229.74 kb), (197.63, 196.66, 229.60 kb) and (197.63, 196.66, 229.54 kb) in fp32, fp16, fp8 respectively.

In fp32, without tcgen05, the GEMM is decomposed into smaller warp-level Tensor Core MMA fragment instructions. While with tcgen05.mma, one thread can launch a tiled mma operation. Furthermore, fp32, Tilelang is unable to use TMEM for the matmul accumulator, creating more warp-level working that the HMMA matmul already needs. In fp16 and fp8, all DSLs use 256x256x64 and 256x256x128 tiles respectively. However, TileLang's lower tensor utilization implies that the data pipeline to cores is not fed fast enough, possibly because of TileLang not using TMA loads. However, this is not true because the bandwidth is not saturated. TMA is hardware movement operation, while TileLang uses cp.async, address arithmetic and shared-memory stores. CuTile has 3.86 and 3.77 issues active % in fp16 and fp8, while Triton has 6.16 and 5.98. TileLang has 11.54 and 9.99 possibly a byproduct of cp.async which is warp-level work rather than TMA loads. TileLang has 4.32x the instruction count than Triton in fp32, while CuTile has 0.12x instruction count than Triton. In fp16 it is 2.80x and 0.54x and in fp8 it is 2.50x and 0.52x. In all dtypes, TileLang issues the most instructions. For fp32, it is almost certainly because of the legacy HMMA which splits the GEMM into smaller warp-level work, while for fp16 and fp8 it is a byproduct of TileLang using cp.async. In fp32, (TL, TR, Cu) LSU activity is 27.31%, 0.06% and 0.10%. In fp16 it is 8.31%, 0.15% and 0.11%. In fp8 it is 7.77%, 0.08% and 0.11%.

### flash_attention

At maximum input case, TileLang-Triton speed-up is 0.62x and CuTile-Triton is 1.10x. Triton selects 128x64, CuTile selects 128x128 and TileLang selects 128x128. Shared Memory is (TL, TR, Cu) is 131.0, 96.6 kb, 224.4 kb. However, TileLang executes 11.95 B instructions, compared to Triton's 10.77 B and CuTile's 4.45 B. TileLang loads 0 bytes via TMA. However, TileLang executes 7.3x CBU instructions than CuTile and 80x CBU instructions than Triton. TileLang is unable to pipeline loads and compute due to compiler problems, and must use LDG and STS instructions via synchronous loading. TileLang uses 233 registers per thread. The greater issue is that TileLang achieves 8 resident warps, Triton achieves 16 and CuTile achieves 12. CuTile has 1 CTA / SM with greater tile sizes and pipelining, while Triton has 2 CTA / SMs leaving 16 warps to hide latency and CTAs synchronize independently so one syncthreads does not stall the other. TileLang has neither. Since the TileLang compiler did not allow pipelining, all the work is serial, and TileLang is stuck at 1 CTA / SM because of its greater register usage and shared memory footprint.

### block_sparse_attention

TileLang uses 62.5 kb, Triton uses 49.2 kb while CuTile uses 221.2 kb. However, both TileLang and CuTile are equally slower than Triton. TileLang-Triton speedup is 0.32x and CuTile-Triton speedup is 0.33x (maximum input case).

Since TileLang does not use as much SMEM as CuTile this is not a simple problem the strategy of using a large tile does not work. Instead, both CuTile and TileLang use tcgen05.mma while Triton stays on legacy HMMA. TileLang continues to load 0 bytes through TMA and both TileLang and CuTile respectively execute 1.93x and 2.14x instructions more than Triton. For non-mma instructions, TileLang has 2.27x and CuTile has 2.51x. Therefore, for both CuTile and TileLang the dominant issue is the additional shared memory staging, synchronization, and sparse block traversal, rather than the choice of MMA instruction family

### linear_self_attention

The column reduction takes 29.4% of total NCU time in TileLang, 57% in Triton, and 53% in CuTile. For this kernel, TileLang takes 245.3us, Triton takes 114.8us and CuTile takes 258.7us. CuTile has 1.75x instructions as Triton, while TileLang has 2.83x instructions compared to Triton.

In the KV GEMM, only CuTile uses tcgen05, while TileLang and Triton both use legacy HMMA at identical counts, and TileLang loads 0 bytes via TMA. TileLang executes 3.11x instructions of Triton. In the output GEMM TileLang is alone in using legacy HMMA, while Triton and CuTile both use tcgen05. TileLang takes 8.02x Triton's time and 2.69 CuTile's time. TileLang does not use cp.async or TMA but rather loads data through LDG and STS. TileLang's LSU works with 16.7x wavefronts compared to Triton, and its Tensor Pipe is at 1.05% compared to Triton's 8.49%.

### flash_decode

CuTile requires 3.72x Triton's latency and 3.75x TileLang's latency (maximum input case). All DSLs use no shared memory, TMA, or tensor core instructions. TileLang executes 0.85x instructions as Triton. All kernels launch 16 CTAs.

### streamk_matmul

Using geomean latency on all cases, TileLang requires 1.54x latency than Triton and 0.78x than CuTile (geomean latency). However, the geomean latency hides patterns across input size and dtype. In fp32, similar to the basic matmul case, TileLang uses legacy HMMA and is only faster than CuTile in the five smallest input cases, all at or below 11.3M output elements; beyond that CuTile is ahead in every case, reaching 2.61x at the maximum input. In fp16/bf16, TileLang is faster than CuTile in 32 of 40 cases, but the gap closes as the output grows — from 0.24x at 5.2M elements to 0.88x at 58.7M — and CuTile overtakes TileLang above 90.2M elements, ending at 1.24x in the maximum input case.

The two backends fail in different kernels, which is why they cross. CuTile's deficit is in first_wave, whose share of the output tiles falls from 54% at the smallest inputs to 1.9% at the largest, so CuTile improves monotonically with input size, from 3.16x Triton at m=1024 to 1.47x at m=8192 in fp16/bf16. TileLang's deficit is in full_tiles, which is an ordinary matmul and dominates as the input grows, so TileLang degrades monotonically, from 1.11x to 1.39x over the same range. In fp32 TileLang's curve is lifted to 2.02x-2.35x because the legacy HMMA fallback applies to every tile at every size, which moves the crossover down from 90.2M to 14.7M elements.
