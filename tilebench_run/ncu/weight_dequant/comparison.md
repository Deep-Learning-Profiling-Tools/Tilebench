# NCU Comparison: weight_dequant

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 4}` |
| bf16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 4}` |
| fp32 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 68.96 us | 70.69 % | 70.69 % | 42.97 % | 34.88 % | 32.04 % | 5.42 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp16 | cutile | 207.84 us | 30.33 % | 23.58 % | 31.15 % | 11.60 % | 82.17 % | 1.81 Tbyte/s | 128 | 32 register/thread | 12.30 Kbyte/block | 0 byte/block | 16 block / 17 block |
| bf16 | triton | 68.83 us | 70.84 % | 70.84 % | 42.76 % | 34.97 % | 32.75 % | 5.43 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 206.34 us | 30.73 % | 23.70 % | 31.40 % | 11.67 % | 84.08 % | 1.82 Tbyte/s | 128 | 32 register/thread | 12.30 Kbyte/block | 0 byte/block | 16 block / 17 block |
| fp32 | triton | 118.78 us | 86.29 % | 86.29 % | 45.64 % | 41.17 % | 22.70 % | 6.62 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 201.57 us | 51.19 % | 51.19 % | 47.53 % | 24.47 % | 82.62 % | 3.93 Tbyte/s | 128 | 32 register/thread | 8.20 Kbyte/block | 0 byte/block | 16 block / 17 block |

## Key findings (auto-derived)

- **fp16**: Triton is **3.01× faster** (69.0 µs vs 207.8 µs).
- **bf16**: Triton is **3.00× faster** (68.8 µs vs 206.3 µs).
- **fp32**: Triton is **1.70× faster** (118.8 µs vs 201.6 µs).

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
