# NCU Comparison: top_k_selection

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'N': 1048576, 'k': 1024}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'block': 2048, 'occupancy': 4, 'K2': 1024}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 197.98 us | 67.76 % | 1.77 % | 84.41 % | 0.86 % | 29.37 % | 135.69 Gbyte/s | 256 | 48 register/thread | 0 byte/block | 8.19 Kbyte/block | 5 block / 11 block |
| fp32 | cutile | 135.44 us | 59.23 % | 3.20 % | 79.13 % | 1.52 % | 28.07 % | 245.08 Gbyte/s | 128 | 72 register/thread | 8.20 Kbyte/block | 0 byte/block | 7 block / 14 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp32 | cutile | 1/10 | 26.46 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 2/10 | 15.68 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 3/10 | 12.22 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 4/10 | 12.00 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 5/10 | 11.78 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 6/10 | 11.74 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 7/10 | 11.58 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 8/10 | 11.23 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 9/10 | 11.26 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | cutile | 10/10 | 11.49 us | `block_topk_kernel_b2048_Kt1_A1f32_1i16t1_p16_A2f32` |
| fp32 | triton | 1/10 | 45.82 us | `block_topk_kernel` |
| fp32 | triton | 2/10 | 26.24 us | `block_topk_kernel` |
| fp32 | triton | 3/10 | 16.26 us | `block_topk_kernel` |
| fp32 | triton | 4/10 | 16.10 us | `block_topk_kernel` |
| fp32 | triton | 5/10 | 15.87 us | `block_topk_kernel` |
| fp32 | triton | 6/10 | 15.84 us | `block_topk_kernel` |
| fp32 | triton | 7/10 | 15.49 us | `block_topk_kernel` |
| fp32 | triton | 8/10 | 15.26 us | `block_topk_kernel` |
| fp32 | triton | 9/10 | 15.87 us | `block_topk_kernel` |
| fp32 | triton | 10/10 | 15.23 us | `block_topk_kernel` |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.46× faster** (135.4 µs vs 198.0 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.49 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
