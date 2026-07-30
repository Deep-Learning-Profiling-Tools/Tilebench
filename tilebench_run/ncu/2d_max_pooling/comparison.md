# NCU Comparison: 2d_max_pooling

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |
| bf16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |
| fp32 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 197.60 us | 57.82 % | 33.67 % | 59.13 % | 19.85 % | 79.61 % | 2.58 Tbyte/s | 128 | 44 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| fp16 | cutile | 287.97 us | 87.06 % | 27.11 % | 88.36 % | 17.36 % | 81.81 % | 2.08 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| bf16 | triton | 200.13 us | 57.02 % | 33.24 % | 58.40 % | 19.59 % | 87.94 % | 2.55 Tbyte/s | 128 | 45 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| bf16 | cutile | 288.80 us | 86.91 % | 27.03 % | 88.38 % | 17.28 % | 81.67 % | 2.07 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| fp32 | triton | 218.59 us | 80.79 % | 61.68 % | 82.54 % | 35.92 % | 66.24 % | 4.73 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 2.05 Kbyte/block | 12 block / 33 block |
| fp32 | cutile | 299.01 us | 89.06 % | 48.86 % | 90.44 % | 26.73 % | 78.02 % | 3.75 Tbyte/s | 128 | 64 register/thread | 2.06 Kbyte/block | 0 byte/block | 8 block / 20 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.46× faster** (197.6 µs vs 288.0 µs).
- **bf16**: Triton is **1.44× faster** (200.1 µs vs 288.8 µs).
- **fp32**: Triton is **1.37× faster** (218.6 µs vs 299.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
