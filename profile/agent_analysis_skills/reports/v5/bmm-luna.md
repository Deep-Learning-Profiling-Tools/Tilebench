# Batched Matmul: fp16, BATCH=32, M=N=K=640

## Benchmark and Capture Comparison

The unique CSV row is line 59 of `results/csv/batched_matmul_autotune.csv`. CSV values are benchmark latency in milliseconds; NCU kernel duration is separately reported below in microseconds. CSV ratios use TileLang latency divided by the named reference. Torch is included as the fastest non-target CSV baseline; among the three DSL backends, Triton is fastest.

| Backend | Selected configuration | CSV latency | TileLang / backend | NCU duration | NCU grid / block | Registers / shared memory per block | Achieved active warps | Tensor pipe active | DRAM read / write | Static SASS instructions (NOP; non-NOP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TileLang | BM128 BN128 BK64, 128 threads, 2 stages, group 1 | 0.0515 ms | 1.000x | 49.248 us | 800 / 128 | 137 / 67,584 B | 6.13% | 16.02% | 31,052,288 / 3,795,968 B | 1,015 (49; 966) |
| Triton | BM128 BN128 BK32, 4 warps, 4 stages, group 1 | 0.0293 ms | 1.758x slower | 29.952 us | 800 / 128 | 135 / 66,608 B | 15.75% | 27.07% | 28,331,264 / 1,683,968 B | 630 (15; 615) |
| cuTile | TM128 TN128 TK64, occupancy hint 4, group 1 | 0.0318 ms | 1.619x slower | 29.920 us | 800 / 256 | 64 / 50,452 B | 28.21% | 26.45% | 25,073,664 / 1,592,832 B | 1,104 (27; 1,077) |
| Torch (CSV only) | Not captured by supplied NCU reports | 0.0233 ms | 2.210x slower | — | — | — | — | — | — | — |

The stored `speedup_tilelang=0.45` is consistent, within printed precision, with `torch_ms / tilelang_ms` (0.4524); it is not a TileLang-over-Torch speedup. The fastest baseline across all supplied CSV columns is Torch. The NCU capture contains one range and one kernel action for each DSL backend, each with 800 CTAs. The reports identify the same chip code 416 and compute capability 10.0. Source stack queried from `profile/env.sh` was PyTorch 2.10.0+cu130, CUDA runtime 13.0, Triton 3.6.0, cuda-tile 1.3.0, and TileLang 0.1.11.

## Likely Mechanism

The most supported explanation is weaker feeding/scheduling of the tensor core work in this TileLang configuration, with resource use and memory movement as plausible contributors. In the supplied implementation snapshot, TileLang stages A/B through shared tiles with `T.copy`, then uses a pipelined `T.gemm` and a tensor-memory accumulator for fp16 (`evidence/implementations/impl_tilelang.py:105-127`). Its SASS has 48 `LDGSTS.E.BYPASS.128` instructions and no `UTMALDG.3D` opcode; Triton’s descriptor path (`evidence/implementations/impl_triton.py:32-38,101-105`) has 10 static `UTMALDG.3D` instructions. Consistent with the different paths, TileLang’s captured tensor-pipe activity is 16.02%, below Triton’s 27.07% and cuTile’s 26.45%, while its achieved active-warps metric is 6.13%, below 15.75% and 28.21% respectively (`targeted_<backend>.ncu-rep`, range 0, action 0). TileLang also reports the largest DRAM read and write byte sums among these captures.

The resource counters do not isolate a single cause. TileLang and Triton have similar per-block registers (137 vs 135) and shared memory (67,584 vs 66,608 B); both report register and shared-memory residency limits of three blocks, while cuTile reports four. These limits make resource pressure plausible but do not explain the large TileLang/Triton achieved-warp difference by themselves. The static SASS counts also do not track latency monotonically: cuTile has the most static instructions yet is fastest in its NCU capture. Thus the load path and low tensor-pipe/warp activity support an underfed tensor-core hypothesis, not a proven compiler defect or a one-counter causal verdict.

## Alternatives and Limits

- NCU replay duration (49.248 vs 29.952/29.920 us) is diagnostic and differs from CSV benchmark latency (51.5 vs 29.3/31.8 us). Keep those ratios separate; the CSV remains the performance source.
- The captures are targeted kernel-replay reports for one action, not full-operation timing. They omit host launch cost, and no staged breakdown is present.
- NCU records include higher TileLang DRAM bytes, but the supplied evidence cannot tell whether this reflects transaction efficiency, cache state, or another generated-code effect. It is supporting evidence only.
- The selected case is exactly tile divisible at 640 for 128-wide M/N tiles and generates 800 blocks. A tail effect is not indicated for this case.
- Static SASS counts include NOPs and do not represent dynamic instruction counts. The listings parsed without warnings; no source-line correlation or dynamic memory transaction details are available here.

To resolve the remaining causal uncertainty, collect matched memory-sector/transaction and stall metrics plus per-kernel occupancy/resource data for these same selected configurations, and inspect the compiler-generated PTX/SASS-to-source mapping. No new profile was run for this analysis.
