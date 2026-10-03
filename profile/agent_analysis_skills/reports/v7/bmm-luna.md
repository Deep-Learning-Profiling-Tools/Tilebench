# Batched Matmul: fp16, M=640

## Finding

The selected TileLang winner is 2.21x slower than the fastest non-TileLang CSV backend (Torch), 1.76x slower than Triton, and 1.62x slower than cuTile. The benchmark row reports 51.5 us for TileLang, 23.3 us for Torch, 29.3 us for Triton, and 31.8 us for cuTile. These are CSV benchmark latencies; Torch has no NCU capture here.

The most plausible operator-level explanation is a less productive, synchronization and staging limited execution schedule in TileLang's tensor-memory path. This remains an inference: the captures show materially lower achieved tensor-pipe activity and more DRAM traffic, while the implementation uses explicit synchronization around tensor-memory GEMM. They do not contain stall or timeline measurements that isolate synchronization as the cause.

| Backend | Benchmark ms | NCU duration ns | Tile/config | Registers/thread | Shared bytes/block | Warp instructions | DRAM read/write bytes | Tensor-pipe active |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| TileLang | 0.0515 | 49,248 | 128x128x64, 128 threads, 2 stages | 137 | 67,584 | 4,011,200 | 31,052,288 / 3,795,968 | 16.02% |
| Triton | 0.0293 | 29,952 | 128x128x32, 4 warps, 4 stages | 135 | 66,608 | 5,448,800 | 28,331,264 / 1,683,968 | 27.07% |
| cuTile | 0.0318 | 29,920 | 128x128x64, occupancy hint 4 | 64 | 50,452 | 1,521,177 | 25,073,664 / 1,592,832 | 26.45% |
| Torch | 0.0233 | not captured | CSV baseline | n/a | n/a | n/a | n/a | n/a |

The comparison case is cubic M=N=K=640 with BATCH=32 (the operator config defines the same M on all three matrix axes). TileLang and Triton use an 800-CTA 5x5x32 grid; cuTile also launches 800 CTAs, in a flattened 25x32 grid. All three NCU reports are from the same manifest commit (`b31f41c38c596bb29d85f66552ef13106e9087df`), each with one kernel range and one action on the same 148-SM device. Captures use the Python path configured by `profile/env.sh`. NCU durations are diagnostic kernel-only durations and must not be substituted for the CSV latency.

## Mechanism and Alternatives

For fp16, TileLang allocates a tensor-memory accumulator and barrier, then calls `T.gemm(..., mbar=...)` and `T.sync_threads()` in its pipelined K loop; after the loop it synchronizes and copies the accumulator back to a fragment before storing ([implementation](../evidence/implementations/impl_tilelang.py), lines 109-124). Triton uses descriptor loads, `tl.dot`, and a descriptor store ([implementation](../evidence/implementations/impl_triton.py), lines 62-75); cuTile uses load, `ct.mma`, and store ([implementation](../evidence/implementations/impl_cutile.py), lines 52-72). The TileLang NCU capture has lower tensor-pipe activity than both alternatives (16.02% versus 27.07% and 26.45%) and lower overall SM throughput (19.71% versus 36.11% and 30.36%). That is consistent with less useful overlap or issue through the tensor path.

This is not explained by larger dynamic instruction count than Triton: `smsp__inst_executed.sum` is 4.0112M for TileLang and 5.4488M for Triton. Nor does the resource ceiling distinguish TileLang from Triton: `launch__occupancy_limit_registers` and `launch__occupancy_limit_shared_mem` both report 3 blocks/SM for each; TileLang's achieved active warps are lower (6.13% versus 15.75%), but achieved activity is not a residency bound. cuTile's ceiling is 4 blocks/SM on those counters, with 28.21% achieved active warps.

TileLang also records 9.6% more DRAM reads and 2.25x the DRAM writes of Triton; versus cuTile it records 23.8% more reads and 2.38x the writes. The counters count physical DRAM bytes and do not reveal transaction efficiency or prove redundant logical work. Static SASS lists contain `UTCHMMA` in all three backends (24 TileLang, 4 Triton, 40 cuTile); TileLang additionally has 48 `LDGSTS.E.BYPASS.128`, 8 `STG.E.ENL2.256`, 10 `BAR.SYNC.DEFER_BLOCKING` and 3 `SYNCS.PHASECHK.TRANS64` instructions. These are static listing counts, including NOPs in total instruction counts, not dynamic execution counts. Triton's static listing has 32 `BAR.SYNC.DEFER_BLOCKING`; therefore barrier presence/count alone does not establish a greater dynamic synchronization cost for TileLang.

The strongest competing explanation is the higher physical DRAM traffic, particularly writes; absent sector/request counters, it cannot be separated from the tensor-memory staging schedule. The selected TileLang configuration uses K=64 and 2 stages versus Triton's K=32 and 4 stages, so it has fewer K iterations but may expose less pipeline overlap. The NCU captures have no issue-stall breakdown, CTA timing distribution, or timeline, and cannot quantify how much latency each effect contributes. Exact synchronization stalls, memory transaction inefficiency, and latency shares remain unresolved.

## Capture and Configuration

Manifest configs: TileLang `{BLOCK_SIZE_M:128, BLOCK_SIZE_N:128, BLOCK_SIZE_K:64, GROUPSIZE:1, threads:128, num_stages:2}`; Triton `{BLOCK_SIZE_M:128, BLOCK_SIZE_N:128, BLOCK_SIZE_K:32, GROUPSIZE:1, num_warps:4, num_stages:4}`; cuTile `{tile_m:128, tile_n:128, tile_k:64, occupancy:4, group_size:1}`. Config source is `results/logs/autotune_logs/batched_matmul_autotune.json`. NCU ratios (TileLang/Triton 1.64x; TileLang/cuTile 1.65x) are separate from the CSV ratios above.
