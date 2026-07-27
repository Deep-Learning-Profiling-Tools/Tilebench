# Matrix Transpose NCU Run

This run compares the tuned FP16 and INT8 transpose kernels across TileLang,
Triton, and cuTile at the largest benchmark shape.

- Shape: `4096 x 20480`
- Dtypes: `fp16`, `int8`
- Configs: recorded autotune winners from
  `results/runs/20260720_045231_b200tuned_resume/operators/matrix_transpose.json`
- Hardware: NVIDIA B200, SM 10.0

The harness performs JIT compilation and three warmup launches before entering
the CUDA profiler region, where it launches exactly one transpose kernel.
