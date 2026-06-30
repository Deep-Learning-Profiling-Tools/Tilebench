# NCU Comparison: 1d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'kernel_size': 127, 'input_size': 20000000}` | `{'BLOCK_SIZE': 512, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'kernel_size': 127, 'input_size': 20000000}` | `{'BLOCK_SIZE': 512, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 404.42 us | 91.65 % | 1.38 % | 92.75 % | 1.38 % | 89.59 % | 105.93 Gbyte/s | 256 | 31 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 583.17 us | 51.57 % | 0.95 % | 53.29 % | 0.95 % | 78.06 % | 73.03 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 588.90 us | 98.38 % | 2.61 % | 99.32 % | 1.79 % | 59.72 % | 200.58 Gbyte/s | 256 | 32 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 452.86 us | 68.65 % | 3.32 % | 70.16 % | 2.21 % | 78.85 % | 254.55 Gbyte/s | 128 | 46 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.44× faster** (404.4 µs vs 583.2 µs).
- **fp32**: cuTile is **1.30× faster** (452.9 µs vs 588.9 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
