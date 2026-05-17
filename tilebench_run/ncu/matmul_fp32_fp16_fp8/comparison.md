# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `(default)` | `(default)` |
| fp16 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `(default)` | `(default)` |
| fp8_e4m3fn | `{'M': 4096, 'N': 4096, 'K': 20480}` | `(default)` | `(default)` |
| fp8_e5m2 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `(default)` | `(default)` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 6250.00 us | 90.13 % | 4.73 % | 91.46 % | 14.29 % | 11.90 % | 363.00 Gbyte/s | 256 | 124 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp32 | cutile | 974.62 us | 86.14 % | 30.42 % | 88.33 % | 64.07 % | 90.70 % | 2.33 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 675.84 us | 56.25 % | 17.70 % | 61.45 % | 48.53 % | 63.49 % | 1.36 Tbyte/s | 256 | 90 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp16 | cutile | 510.56 us | 83.86 % | 28.98 % | 86.95 % | 60.48 % | 88.39 % | 2.22 Tbyte/s | 256 | 255 register/thread | 229.57 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 70.85 us | 74.30 % | 74.30 % | 20.88 % | 41.02 % | 66.23 % | 5.70 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e4m3fn | cutile | 70.91 us | 74.18 % | 74.18 % | 20.90 % | 40.95 % | 66.39 % | 5.69 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | triton | 71.10 us | 74.08 % | 74.08 % | 20.85 % | 40.95 % | 67.92 % | 5.68 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | cutile | 71.46 us | 73.68 % | 73.68 % | 20.75 % | 40.70 % | 67.72 % | 5.65 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **6.41× faster** (974.6 µs vs 6250.0 µs).
- **fp16**: cuTile is **1.32× faster** (510.6 µs vs 675.8 µs).
- **fp8_e4m3fn**: Triton is **1.00× faster** (70.8 µs vs 70.9 µs).
- **fp8_e5m2**: Triton is **1.01× faster** (71.1 µs vs 71.5 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp8_e4m3fn / cutile** — Compute and Memory are well-balanced
- **fp8_e4m3fn / triton** — Compute and Memory are well-balanced
- **fp8_e5m2 / cutile** — Compute and Memory are well-balanced
- **fp8_e5m2 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_fp8_e4m3fn.ncu-rep`
- `cutile_fp8_e5m2.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_fp8_e4m3fn.ncu-rep`
- `triton_fp8_e5m2.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
