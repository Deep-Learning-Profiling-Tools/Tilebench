# NCU Comparison: 1d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

**Operator:** multi-channel Conv1d forward (batch=1, C_in=C_out=128, k=3, pad=1) — **implicit GEMM** on both
backends: one CTA computes a (BLOCK_M x BLOCK_OUT) output tile, contracting over C_in*k = 384
via Tensor-Core MMA (Triton `tl.dot(input_precision="tf32")`, cuTile `ct.mma` with tfloat32/native
cast). torch reference: native-dtype cuDNN (fp16 -> fp16 TC, fp32 -> TF32 where cuDNN supports it).
Same algorithm, same tiling scheme, same autotune tile space ({32,64,128}x{16,32,64}x{64,128}) on
both backends; DSL-specific knobs differ (Triton num_warps/num_stages vs cuTile occupancy).

## Autotune winners (sweep-max, L=2621440)

| dtype | Triton | cuTile |
|---|---|---|
| fp16 | `{'BLOCK_SIZE_BATCH_LENGTH': 128, 'BLOCK_SIZE_IN_FEAT': 32, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 8, 'num_stages': 2}` | `{'block_bl': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 8}` |
| fp32 | `{'BLOCK_SIZE_BATCH_LENGTH': 64, 'BLOCK_SIZE_IN_FEAT': 16, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 4, 'num_stages': 3}` | `{'block_bl': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 8}` |

## Headline (autotune-best @ sweep-max; times from the sweep CSV, ms from NCU)

| dtype | Backend | NCU duration | Tensor pipe % | UTCMMA (tcgen05) | Occupancy % | SM % |
|---|---|---|---|---|---|---|
| fp16 | triton | 2.5 ms | 4.6 | 491,520 | 24.6 | 64.0 |
| fp16 | cutile | 4.5 ms | 10.3 | 0 | 49.6 | 77.4 |
| fp32 | triton | 3.1 ms | 15.1 | 1,966,080 | 24.6 | 66.4 |
| fp32 | cutile | 9.7 ms | 9.5 | 0 | 49.7 | 50.2 |

## Sweep-max latencies (autotune CSV)

| dtype | torch_ms | triton_ms | cutile_ms | C/T |
|---|---|---|---|---|
| fp16 | 1.3128 | 2.5157 | 4.4981 | 1.7880 |
| fp32 | 1.7363 | 3.0664 | 10.6284 | 3.4661 |

## Key findings

- **Both backends run on Tensor Cores** (tensor pipe active on every report). Triton compiles to
  tcgen05 `UTCMMA`; cuTile picks tcgen05 or legacy HMMA depending on the tile shape (UTCMMA=0
  rows still show a busy hmma sub-pipe).
- Both DSLs trail torch's cuDNN, which uses dedicated implicit-GEMM conv kernels; the DSL kernels
  spend most cycles on im2col index arithmetic (ALU-bound, SM% 50-77 with low tensor%), identically
  on both sides — so the Triton-vs-cuTile delta isolates DSL codegen, which is the benchmark's goal.

## Reports

- `triton_fp16.ncu-rep`, `triton_fp32.ncu-rep`, `cutile_fp16.ncu-rep`, `cutile_fp32.ncu-rep`
