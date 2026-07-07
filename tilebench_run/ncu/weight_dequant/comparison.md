# NCU Comparison: weight_dequant

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| bf16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| fp32 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 4096, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 62.72 us | 77.21 % | 77.21 % | 46.09 % | 38.34 % | 31.74 % | 5.92 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 188.61 us | 26.09 % | 26.09 % | 26.37 % | 12.77 % | 87.86 % | 2.00 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| bf16 | triton | 63.26 us | 76.65 % | 76.65 % | 46.45 % | 38.03 % | 31.88 % | 5.87 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 188.58 us | 25.99 % | 25.99 % | 26.40 % | 12.76 % | 87.75 % | 1.99 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| fp32 | triton | 117.02 us | 87.46 % | 87.46 % | 45.86 % | 41.77 % | 22.94 % | 6.71 Tbyte/s | 256 | 28 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 224.45 us | 46.19 % | 46.19 % | 39.81 % | 21.86 % | 68.68 % | 3.54 Tbyte/s | 128 | 112 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |

## Key findings (auto-derived)

- **fp16**: Triton is **3.01× faster** (62.7 µs vs 188.6 µs).
- **bf16**: Triton is **2.98× faster** (63.3 µs vs 188.6 µs).
- **fp32**: Triton is **1.92× faster** (117.0 µs vs 224.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Compute is more heavily utilized than Memory
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
