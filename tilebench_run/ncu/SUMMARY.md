# NCU Sweep Summary — all operators

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** autotune-winner cfg at sweep-max input case, `--set full --import-source on`, `--launch-skip 3 --launch-count 1`

Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the `<backend>_<dtype>.ncu-rep` files in that directory.

## Headline duration table (µs)

| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |
|---|---|---:|---:|---:|
| leaky_relu | fp16 | 26.8 | 27.2 | 1.02× |
| leaky_relu | bf16 | 26.4 | 26.7 | 1.01× |
| leaky_relu | fp32 | 55.8 | 56.1 | 1.01× |
| dropout | fp16 | 18.3 | 18.6 | 1.02× |
| dropout | bf16 | 18.1 | 19.1 | 1.06× |
| dropout | fp32 | 34.9 | 34.5 | 0.99× |
| 2d_max_pooling | fp16 | 196.9 | 286.2 | 1.45× |
| 2d_max_pooling | bf16 | 199.5 | 286.2 | 1.43× |
| 2d_max_pooling | fp32 | 218.5 | 297.9 | 1.36× |
| 3d_conv | fp16 | 13850.0 | 35670.0 | 2.58× |
| 3d_conv | fp32 | 18200.0 | 33080.0 | 1.82× |
| argmax | fp16 | 30.4 | 31.2 | 1.03× |
| argmax | fp32 | 41.1 | 31.6 | 0.77× |
| bitonic_sort | fp16 | 8116.7 | 8107.6 | 1.00× |
| bitonic_sort | fp32 | 9277.0 | 9114.4 | 0.98× |
| cross_entropy | fp16 | 9.8 | 14.4 | 1.46× |
| cross_entropy | fp32 | 10.1 | 13.9 | 1.38× |
| dropout | fp16 | 18.2 | 19.1 | 1.05× |
| dropout | bf16 | 18.5 | 18.9 | 1.02× |
| dropout | fp32 | 34.4 | 34.5 | 1.00× |
| cross_entropy | fp16 | 9.9 | 15.0 | 1.53× |
| cross_entropy | fp32 | 10.5 | 13.9 | 1.33× |
| gaussian_blur | fp16 | 1030.0 | 1510.0 | 1.47× |
| gaussian_blur | fp32 | 1280.0 | 1290.0 | 1.01× |
| jacobi_stencil_2d | fp16 | 155.6 | 134.9 | 0.87× |
| jacobi_stencil_2d | bf16 | 155.6 | 134.3 | 0.86× |
| jacobi_stencil_2d | fp32 | 216.7 | 179.5 | 0.83× |
| layernorm | fp16 | 20.0 | 35.3 | 1.76× |
| layernorm | bf16 | 20.8 | 37.2 | 1.79× |
| layernorm | fp32 | 32.7 | 33.6 | 1.03× |
| matrix_copy | fp16 | 17.8 | 18.4 | 1.03× |
| matrix_copy | bf16 | 18.0 | 18.3 | 1.02× |
| matrix_copy | fp32 | 29.4 | 27.9 | 0.95× |
| matrix_copy | int8 | 10.8 | 11.3 | 1.04× |
| matrix_transpose | fp16 | 50.5 | 50.1 | 0.99× |
| matrix_transpose | bf16 | 50.1 | 50.6 | 1.01× |
| matrix_transpose | fp32 | 99.1 | 99.9 | 1.01× |
| matrix_transpose | int8 | 28.9 | 27.7 | 0.96× |
| mean_reduction | fp16 | 16.7 | 19.7 | 1.18× |
| mean_reduction | bf16 | 16.7 | 19.6 | 1.17× |
| mean_reduction | fp32 | 29.9 | 30.2 | 1.01× |
| moe_topk_gating | fp16 | 14.8 | 29.6 | 2.00× |
| moe_topk_gating | bf16 | 14.8 | 29.3 | 1.98× |
| moe_topk_gating | fp32 | 14.9 | 29.7 | 1.99× |
| mul2 | fp16 | 14.7 | 15.7 | 1.07× |
| mul2 | bf16 | 13.5 | 15.4 | 1.13× |
| mul2 | fp32 | 22.4 | 22.7 | 1.01× |
| mul2 | int8 | 9.7 | 11.2 | 1.15× |
| relu | fp16 | 14.8 | 13.0 | 0.87× |
| relu | bf16 | 14.9 | 13.1 | 0.88× |
| relu | fp32 | 22.1 | 22.6 | 1.02× |
| relu | int8 | 10.8 | 14.0 | 1.29× |
| reverse_array | fp16 | 16.4 | 15.2 | 0.93× |
| reverse_array | bf16 | 16.9 | 17.0 | 1.00× |
| reverse_array | fp32 | 21.9 | 22.0 | 1.00× |
| reverse_array | int8 | 14.5 | 12.6 | 0.87× |
| swiglu | fp16 | 70.0 | 98.6 | 1.41× |
| swiglu | bf16 | 72.5 | 99.2 | 1.37× |
| swiglu | fp32 | 138.3 | 143.0 | 1.03× |
| vector_add | fp16 | 18.5 | 18.6 | 1.00× |
| vector_add | bf16 | 17.9 | 17.8 | 0.99× |
| vector_add | fp32 | 34.2 | 35.2 | 1.03× |
| vector_add | int8 | 12.2 | 13.5 | 1.11× |
| weight_dequant | fp16 | 62.0 | 187.4 | 3.03× |
| weight_dequant | bf16 | 62.9 | 187.5 | 2.98× |
| weight_dequant | fp32 | 118.3 | 194.6 | 1.65× |
| 1d_conv | fp16 | 2520.0 | 4510.0 | 1.79× |
| 1d_conv | fp32 | 3080.0 | 5320.0 | 1.73× |
| 2d_conv | fp16 | 552.0 | 797.2 | 1.44× |
| 2d_conv | fp32 | 626.5 | 934.1 | 1.49× |
| batch_normalization | fp16 | 32.3 | 54.7 | 1.69× |
| batch_normalization | bf16 | 33.1 | 55.0 | 1.66× |
| batch_normalization | fp32 | 61.0 | 52.2 | 0.86× |
| batched_matmul | fp16 | 29.8 | 30.3 | 1.02× |
| batched_matmul | bf16 | 29.9 | 29.8 | 1.00× |
| batched_matmul | fp32 | 49.5 | 99.3 | 2.01× |
| block_sparse_attention | fp16 | 69.6 | 222.0 | 3.19× |
| dequantize_rowwise | int8 | 21.6 | 22.9 | 1.06× |
| destindex | fp16 | 42.5 | 115.0 | 2.71× |
| destindex | bf16 | 42.4 | 115.3 | 2.72× |
| destindex | fp32 | 85.2 | 126.6 | 1.49× |
| destindex | int8 | 41.5 | 111.6 | 2.69× |
| flash_attention | fp16 | 19340.0 | 17680.0 | 0.91× |
| flash_decode | fp32 | 47.4 | 168.3 | 3.55× |
| fused_activation | fp32 | 47.3 | 48.0 | 1.01× |
| histogramming | int32 | 793.7 | 1436.6 | 1.81× |
| interleave | fp16 | 21.6 | 22.7 | 1.05× |
| interleave | bf16 | 21.5 | 22.8 | 1.06× |
| interleave | fp32 | 44.7 | 45.1 | 1.01× |
| interleave | int8 | 13.1 | 13.1 | 1.00× |
| kl_divergence | fp32 | 94.5 | 89.1 | 0.94× |
| l2_norm | fp16 | 14.1 | 19.1 | 1.35× |
| l2_norm | bf16 | 14.1 | 18.7 | 1.33× |
| l2_norm | fp32 | 27.5 | 30.2 | 1.10× |
| linear_self_attention | fp32 | 415.4 | 2269.6 | 5.46× |
| matmul_fp32_fp16_fp8 | fp32 | 1340.0 | 871.7 | 0.65× |
| matmul_fp32_fp16_fp8 | fp16 | 518.0 | 467.2 | 0.90× |
| matmul_fp32_fp16_fp8 | fp8_e4m3fn | 257.1 | 220.8 | 0.86× |
| matmul_int8 | int8 | 338.0 | 341.1 | 1.01× |
| quantize_global | fp32 | 17.7 | 18.5 | 1.04× |
| radix_sort | int32 | 1331.2 | 2529.9 | 1.90× |
| rmsnorm | fp16 | 18.0 | 21.8 | 1.21× |
| rmsnorm | bf16 | 17.9 | 21.2 | 1.19× |
| rmsnorm | fp32 | 30.4 | 32.4 | 1.07× |
| rope | fp16 | 49.4 | 52.5 | 1.06× |
| rope | fp32 | 101.7 | 106.3 | 1.05× |
| sigmoid | fp16 | 30.7 | 50.9 | 1.66× |
| sigmoid | bf16 | 30.9 | 50.8 | 1.64× |
| sigmoid | fp32 | 56.2 | 61.9 | 1.10× |
| softmax | fp16 | 23.8 | 43.2 | 1.82× |
| softmax | fp32 | 32.6 | 44.5 | 1.36× |
| streamk_matmul | fp16 | 1962.6 | 2220.7 | 1.13× |
| streamk_matmul | bf16 | 2073.8 | 2171.1 | 1.05× |
| streamk_matmul | fp32 | 3628.5 | 3733.4 | 1.03× |
| top_k_selection | fp32 | 200.6 | 140.2 | 0.70× |

## Failed pairs

# NCU sweep failures

- **cross_entropy/fp16/triton** (rc=1, 3.96s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 88, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 76, in main
    out = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/operators/cross_entropy/impl_triton.py", line 74, in run
    num_stages=cfg["num_stages"],
KeyError: 'num_stages'
  ```

- **cross_entropy/fp32/triton** (rc=1, 3.95s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 88, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 76, in main
    out = impl.run(*inputs)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/operators/cross_entropy/impl_triton.py", line 74, in run
    num_stages=cfg["num_stages"],
KeyError: 'num_stages'
  ```

- **relu/int8/triton** (rc=1, 3.34s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 90, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 73, in main
    inputs = GENERATORS[op](**params)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/data/tensors.py", line 72, in generate_relu_inputs
    x = torch.randn(n, dtype=dtype, device=device)
NotImplementedError: "normal_kernel_cuda" not implemented for 'Char'
  ```

- **relu/int8/cutile** (rc=1, 4.6s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 90, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 73, in main
    inputs = GENERATORS[op](**params)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/data/tensors.py", line 72, in generate_relu_inputs
    x = torch.randn(n, dtype=dtype, device=device)
NotImplementedError: "normal_kernel_cuda" not implemented for 'Char'
  ```

- **relu/int8/triton** (rc=1, 3.32s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 90, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 73, in main
    inputs = GENERATORS[op](**params)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/data/tensors.py", line 72, in generate_relu_inputs
    x = torch.randn(n, dtype=dtype, device=device)
NotImplementedError: "normal_kernel_cuda" not implemented for 'Char'
  ```

- **relu/int8/cutile** (rc=1, 4.56s)
  ```
  Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 90, in <module>
    main()
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/ncu_generic_harness.py", line 73, in main
    inputs = GENERATORS[op](**params)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/data/tensors.py", line 72, in generate_relu_inputs
    x = torch.randn(n, dtype=dtype, device=device)
NotImplementedError: "normal_kernel_cuda" not implemented for 'Char'
  ```