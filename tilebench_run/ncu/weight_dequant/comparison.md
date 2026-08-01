# NCU Comparison: weight_dequant

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| bf16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| fp32 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 61.95 us | 78.23 % | 78.23 % | 45.77 % | 38.79 % | 31.77 % | 6.00 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 187.42 us | 26.27 % | 26.27 % | 26.39 % | 12.86 % | 87.70 % | 2.02 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| bf16 | triton | 62.88 us | 77.40 % | 77.40 % | 45.50 % | 38.29 % | 31.62 % | 5.93 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 187.52 us | 26.27 % | 26.27 % | 26.35 % | 12.85 % | 87.83 % | 2.01 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| fp32 | triton | 118.30 us | 86.61 % | 86.61 % | 45.60 % | 41.38 % | 22.78 % | 6.64 Tbyte/s | 256 | 28 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 194.62 us | 53.24 % | 53.24 % | 46.57 % | 25.21 % | 85.57 % | 4.08 Tbyte/s | 128 | 70 register/thread | 16.40 Kbyte/block | 0 byte/block | 7 block / 7 block |

## Key findings (auto-derived)

- **fp16**: Triton is **3.03× faster** (62.0 µs vs 187.4 µs).
- **bf16**: Triton is **2.98× faster** (62.9 µs vs 187.5 µs).
- **fp32**: Triton is **1.65× faster** (118.3 µs vs 194.6 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
