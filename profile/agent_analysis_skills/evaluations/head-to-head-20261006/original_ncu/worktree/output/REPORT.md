# B200 FP32 2d_max_pooling: selected-winner comparison

Analysis date: 2026-10-06. Saved evidence only; no GPU runs or new tuning.
Workload: N=4, C=128, H=W=640, kernel=3, stride=2, padding=1; output H=W=320.
The source configuration defines these defaults at `tilebench/benchmarks/operators/2d_max_pooling/config.yaml:8`.
The input is 838,860,800 bytes and the output 209,715,200 bytes, derived from shape and FP32 size.

## Main result

TileLang and Triton are within the study's 10% parity threshold. cuTile is materially slower.
The strongest supported mechanism is compiler-generated address/mask and output-layout overhead, with cuTile also paying higher register pressure and redundant memory traffic.
This is a comparison of selected implementations, not a proof that one language inherently generates faster pooling.

| Performance source | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| CSV latency, ms | 0.2063 | 0.2257 | 0.2995 |
| CSV latency / TileLang latency | 1.0000 | 1.0940 | 1.4518 |
| Saved NCU duration, us | 211.712 | 227.520 | 306.432 |
| NCU duration / TileLang duration | 1.0000 | 1.0747 | 1.4474 |

CSV ratios above are explicitly `backend_ms / tilelang_ms`; TileLang is 8.60% lower latency than the fastest other backend, Triton, and 31.12% lower than cuTile.
The CSV's ambiguous `triton_vs_cutile=1.3270` means `cutile_ms / triton_ms`, not the reverse; recomputation gives 1.32698.
CSV is the performance source. NCU independently supports the ordering and mechanism; the two timing series are not combined.

## Capture correspondence and limits

Current source is the supplied checkout at 17d2d4f6; winner archive is identified as 9455c0bd.
`evidence/reports.json` identifies dataset revision 21037737b7e371d38f3d029367dc3967d5b31a23; a dataset revision does not establish source identity.
All three SHA-256 hashes match the supplied manifest. Each report contains exactly one range and one action, enumerated in `action_inventory.json`.
The exact kernel names and report identities are retained with every selected metric in `selected_records.json`.
The FP32 H=640 records in both raw winner logs select TileLang `{BLOCK_R:1,BLOCK_C:512,threads:128}`, Triton `{BLOCK_R:1,BLOCK_C:512,num_warps:4}`, and cuTile `{tile_r:4,tile_c:128,occupancy:8}`.
Captured grids are respectively (512,320,1), (512,320,1), and (512,80,3), each with a 128-thread block: precisely the grids implied by those winners.
Captured cuTile's kernel name also contains its kernel/stride/padding/tile constants. The occupancy hint is consistent with, but not proven by, the observed resource limit.
Embedded Triton and cuTile kernel-body ASTs equal the current kernel bodies (`source_correspondence.json`); whole files differ in wrapper/import or tuner settings.
TileLang names `/tmp/tmpk2z7oy3_/tvm_kernels.cu` in line information, but its CUDA content is absent. SASS embeds the expected row/plane strides and output width, supporting correspondence without proving exact source or compiler identity.
No capture compiler version/flags or input contents are inferred from matching geometry. No historical source or external report was consulted.

## Side-by-side causal evidence

All rows below are saved NCU metrics; M denotes millions, MB denotes decimal megabytes. Exact metric names, values, units, report, range, action and kernel are in `selected_records.json`.

| Evidence | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Executed warp instructions (`smsp__inst_executed.sum`), M | 62.2592 | 160.3994 | 234.9466 |
| Registers/thread; allocated registers/thread | 37; 40 | 39; 40 | 64; 64 |
| Register-limited resident blocks/SM | 12 | 12 | 8 |
| Theoretical / achieved active occupancy, % | 75 / 65.22 | 75 / 67.36 | 50 / 46.01 |
| Static / dynamic shared allocation, bytes | 0 / 0 | 0 / 2048 | 2060 / 0 |
| Waves/SM | 92.25 | 92.25 | 103.78 |
| Global load requests, M | 15.2212 | 15.2212 | 14.8685 |
| Global load sectors, M | 122.7520 | 122.7520 | 137.4822 |
| Global load useful bytes/sector | 15.36 | 15.36 | 13.71 |
| DRAM read / write, MB | 839.020 / 251.604 | 839.145 / 251.400 | 922.841 / 254.654 |
| L1 / L2 hit rate, % | 64.60 / 23.86 | 64.59 / 24.55 | 74.49 / 21.14 |
| L1TEX / DRAM throughput, % sustained peak | 61.71 / 67.16 | 77.25 / 62.48 | 86.78 / 50.09 |
| Long-scoreboard samples / total samples | 7058 / 14378 | 9052 / 15260 | 8883 / 20612 |
| Long-scoreboard fraction, % (derived) | 49.09 | 59.32 | 43.10 |

## Why the implementations differ

**TileLang versus Triton:** both algorithms traverse the same window and use the same winner geometry (`impl_tilelang.py:45`, `impl_triton.py:33`). Their identical load request/sector counts and near-identical HBM traffic exclude a meaningful input-traffic advantage as the explanation of this small latency difference.
TileLang specializes shape constants in `T.Tensor` declarations (`impl_tilelang.py:29`); captured SASS forms a base address once, uses immediate load offsets, and removes wholly out-of-output column work. Triton retains runtime H/W/H_out/W_out address and bounds calculations (`impl_triton.py:12`), even for this fixed case.
The Triton SASS epilogue explicitly performs `STSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING`, `LDS.128`, then `STG.E.128`: an output layout conversion absent from TileLang's direct `STG.E` path (`sass_triton.txt`, PCs ending 0xe80-0xf60; `sass_tilelang.txt`, 0x590-0x5e0).
Triton executes 2.576 times TileLang's warp instructions, yet both allocate the same register granularity and have similar occupancy. The extra instructions and rearrangement are real but largely overlap memory service; they do not imply a proportional latency penalty or a significant benchmark win beyond parity.

**cuTile versus both:** `ct.gather` and `ct.store` operate on tensor descriptors (`impl_cutile.py:43`, `impl_cutile.py:46`). Captured SASS repeatedly computes wide addresses and bounds from runtime descriptor data, then rearranges outputs with `STS`, a block barrier, and `LDS.128` before the vector global store (PCs ending 0x1d10-0x1dc0).
This is captured lowering overhead, not an asserted missing language capability. cuTile executes 3.774 times TileLang's warp instructions; its 64-register allocation cuts theoretical occupancy to 50%, versus 75% for the other two, reducing available memory-latency hiding.
Its load sectors rise 12.00% and DRAM reads 9.99% relative to TileLang, even though load requests are slightly lower. Its mapping transfers fewer useful bytes per sector; the 4x128 tile also produces a partially filled last column tile. Descriptor/index costs, access packing and block-local overlap jointly matter; the saved evidence does not isolate each contribution.
cuTile's higher L1 hit rate therefore is not proof of more efficient memory use: total sectors and HBM reads are higher, while L1TEX throughput is nearer peak. The mechanism is excess memory-system work plus expensive lowering and reduced residency, not lower cache hit rate alone.

## Remaining analysis dimensions and priorities

All three have large grids and over 90 waves/SM. Per-SM active-cycle maxima/minima are within 1.8% of their averages, giving no support for a dominant small-grid or gross block-imbalance explanation. No PM sampling metric exists, so the utilization timeline and temporal tail shape cannot be diagnosed.
Long-scoreboard samples are concentrated at load-dependent maximum operations: Triton source line 41, cuTile lines 43-44 and its store preparation, TileLang generated CUDA line 38 / inlined maximum code. `stall_hotspots_*.json` retains exact PCs; these are dependency waits, not automatically the producing load locations.
DRAM throughput of 50-67% and L1TEX throughput of 62-87% rule out the playbook's low-DRAM, purely latency-bound pattern. These are mixed memory-service and lowering costs; sampled percentages are not elapsed-time attribution and cannot rank backend latency alone.
Tensor-pipe activity is zero for all three, as expected for scalar max pooling. Local load/store request counters are zero, excluding observed register spill traffic. Triton/cuTile do have shared bank-conflict counters, but the output conversion and resource costs are better-supported primary explanations than a dominant bank-conflict diagnosis.
Priority 1: investigate specializing cuTile descriptor sizes/strides and simplifying its gather bounds/address path; evidence is the SASS and instruction/register inflation. Priority 2: investigate a direct-store layout that removes the shared output rearrangement, especially in cuTile. Priority 3: improve adjacent-window input reuse or load packing, supported by roughly half-sector useful-byte utilization and long-scoreboard waits in every backend.
NCU rule outputs are archived in `rules_*.json`, including access-pattern, scoreboard and occupancy warnings and their hypothetical speedup estimates. Those overlapping estimates are not additive or measured expected gains; no change or config sweep was performed.

Reproduction of this saved-evidence analysis is in `COMMANDS.md`; full scalar dumps, SASS/PTX listings, source inventories and scripts accompany this report in `output/`.
