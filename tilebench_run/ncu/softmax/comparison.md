# NCU Comparison: softmax

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n_rows': 2048, 'n_cols': 10240}` | `(default)` | `(default)` |
| fp32 | `{'n_rows': 2048, 'n_cols': 10240}` | `(default)` | `(default)` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 27.26 us | 35.27 % | 21.33 % | 44.10 % | 19.53 % | 61.45 % | 1.63 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 44.19 us | 27.08 % | 13.12 % | 31.59 % | 12.42 % | 61.32 % | 1.01 Tbyte/s | 256 | 32 register/thread | 26.81 Kbyte/block | 0 byte/block | 8 block / 8 block |
| fp32 | triton | 36.93 us | 60.83 % | 60.83 % | 43.88 % | 35.70 % | 47.68 % | 4.66 Tbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 49.44 us | 35.98 % | 30.96 % | 41.10 % | 21.57 % | 55.67 % | 2.37 Tbyte/s | 256 | 32 register/thread | 22.99 Kbyte/block | 0 byte/block | 8 block / 8 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.62× faster** (27.3 µs vs 44.2 µs).
- **fp32**: Triton is **1.34× faster** (36.9 µs vs 49.4 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
