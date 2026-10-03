# Argmax N=2048 fp16

## Result

For `M=2048`, `N=2048`, fp16, the printed benchmark row reports TileLang 0.0031 ms (3.1 us), Triton 0.0049 ms and cuTile 0.0048 ms. Among the three available non-TileLang columns (`torch_ms`, `triton_ms`, `cutile_ms`), cuTile is fastest. TileLang latency is 0.646x cuTile, or 35.4% lower; it is 0.633x Triton. The selected row is CSV line 4; the helper result with all three references is preserved in `csv_checks.json`.

| Backend | Selected winning configuration | Loop chunks for N=2048 | CSV latency | NCU duration | Dynamic warp instructions | L1 global-load requests / sectors |
|---|---|---:|---:|---:|---:|---:|
| TileLang | BLOCK_N=2048, 128 threads, 3 stages | 1 | 3.1 us | 5888 ns | 1,187,840 | 8,192 / 262,144 |
| Triton | BLOCK_N=1024, 4 warps, 3 stages | 2 | 4.9 us | 7808 ns | 2,422,784 | 16,384 / 262,144 |
| cuTile | block_n=1024, occupancy=16 | 2 | 4.8 us | 7520 ns | 2,629,632 | 16,384 / 262,144 |

Winning configurations are from the selected fp16 record in `argmax_autotune.json` (lines 46-66); the CSV case and row are in `argmax_autotune.csv` (line 4). Implementations show TileLang's loop is `ceildiv(N, BLOCK_N)` and processes a tile max plus a minimum matching index each iteration ([TileLang lines 42-59](../benchmarks/operators/argmax/impl_tilelang.py:42)). Triton and cuTile use 1024-wide chunks in their selected configuration and perform max/argmax per chunk ([Triton lines 17-27](../benchmarks/operators/argmax/impl_triton.py:17), [cuTile lines 31-42](../benchmarks/operators/argmax/impl_cutile.py:31)). Thus the TileLang winner scans the same row in one loop chunk where both comparison winners use two.

## Mechanism

The evidence supports fewer repeated chunk/reduction iterations as the primary operator-level explanation. All captures use one range and one action (`range_index=0`, `action_index=0`); report inventories contain 365 metric names each. TileLang executes about 49.0% of Triton's and 45.2% of cuTile's `smsp__inst_executed.sum`, while its NCU duration is 24.6% below Triton and 21.7% below cuTile. Shared instruction counts are 65,536 for TileLang versus 106,496 for each other backend. The matching input-sector count is 262,144 for all three, but TileLang has half as many L1 global-load requests. Together, this fits one 2048-wide pass versus two 1024-wide passes, with less repeated reduction/control work rather than less input data.

The work-count ratio is considerably smaller than the CSV latency ratio. NCU duration is a kernel capture and excludes host launch overhead; it is not a substitute for benchmark latency. The capture reports equal grid size (2048), block size (128), and waves per SM (0.8649); register counts are 32/29/32 and shared-memory allocations are 1536/1056/1068 bytes for TileLang/Triton/cuTile. TileLang's achieved active-warps metric is 63.2%, compared with 58.2% and 61.1%; these values do not suggest an occupancy collapse. Its long-scoreboard issue-normalized stall ratio is higher (3.85 versus 2.23/2.22), so scoreboard stalls do not explain TileLang's advantage. These stall ratios are not elapsed-time percentages.

Static imported SASS listings parse without warnings: 160 TileLang, 184 Triton, and 336 cuTile instructions including NOPs. These are static code sizes, not dynamic counts; loop repetition is reflected in NCU's dynamic instruction counter instead. The reports show only 4,608/5,120/7,424 DRAM bytes read under `cache-control none`, far below the 8 MiB fp16 input. Since the report observes identical L1 sector counts and apparently cached data, DRAM-byte differences do not support a global-traffic explanation. The inputs' effective cache residency limits what this capture says about cold-memory behavior.

## Limits and Provenance

This diagnosis is an inference from the selected configurations, implementation structure and matching one-action NCU counters; it does not isolate how much of the measured latency delta comes from loop trip count versus each backend's reduction lowering. Exact compiler-specific attribution is unresolved. The helper's static listing counts also do not establish which instructions execute per loop iteration.

The benchmark CSV belongs to the repository's recorded PDF-era stack. However, this capture manifest records an NCU executable at `/usr/local/cuda-13.2/bin/ncu`, while the supplied study instructions specify CUDA 13.0. The supplied capture metadata does not establish the runtime PyTorch/Triton/cuTile versions. This provenance mismatch limits direct comparison of absolute NCU timing against the CSV; the within-capture relative counters remain diagnostic evidence. No new compile, profile, benchmark, or autotune was run.
