# NCU Comparison: top_k_selection

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'N': 1048576, 'k': 1024}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'block': 4096, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 200.03 us | 67.51 % | 1.21 % | 85.23 % | 0.80 % | 29.27 % | 92.55 Gbyte/s | 256 | 48 register/thread | 0 byte/block | 8.19 Kbyte/block | 5 block / 11 block |
| fp32 | cutile | 156.33 us | 0.33 % | 0.05 % | 33.20 % | 0.09 % | 0.09 % | 3.46 Gbyte/s | 128 | 101 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp32 | cutile | 1/5 | 30.88 us | `block_topk_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0` |
| fp32 | cutile | 2/5 | 29.73 us | `block_topk_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0` |
| fp32 | cutile | 3/5 | 29.70 us | `block_topk_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0` |
| fp32 | cutile | 4/5 | 30.40 us | `block_topk_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0` |
| fp32 | cutile | 5/5 | 35.62 us | `block_topk_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0` |
| fp32 | triton | 1/10 | 45.79 us | `block_topk_kernel` |
| fp32 | triton | 2/10 | 25.98 us | `block_topk_kernel` |
| fp32 | triton | 3/10 | 16.16 us | `block_topk_kernel` |
| fp32 | triton | 4/10 | 16.10 us | `block_topk_kernel` |
| fp32 | triton | 5/10 | 15.87 us | `block_topk_kernel` |
| fp32 | triton | 6/10 | 16.10 us | `block_topk_kernel` |
| fp32 | triton | 7/10 | 16.06 us | `block_topk_kernel` |
| fp32 | triton | 8/10 | 15.97 us | `block_topk_kernel` |
| fp32 | triton | 9/10 | 16.06 us | `block_topk_kernel` |
| fp32 | triton | 10/10 | 15.94 us | `block_topk_kernel` |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.28× faster** (156.3 µs vs 200.0 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
