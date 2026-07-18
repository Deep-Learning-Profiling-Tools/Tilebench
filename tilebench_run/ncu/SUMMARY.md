# NCU Sweep Summary — all operators

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** autotune-winner cfg at sweep-max input case, `--set full --import-source on`, `--launch-skip 3 --launch-count 1`

Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the `<backend>_<dtype>.ncu-rep` files in that directory.

## Headline duration table (µs)

| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |
|---|---|---:|---:|---:|
| 1d_conv | fp16 | 2520.0 | 4510.0 | 1.79× |
| 1d_conv | fp32 | 3080.0 | 9710.0 | 3.15× |
| 2d_conv | fp16 | 552.0 | 797.2 | 1.44× |
| 2d_conv | fp32 | 626.5 | 934.1 | 1.49× |
| 2d_max_pooling | fp16 | 197.6 | 288.0 | 1.46× |
| 2d_max_pooling | bf16 | 200.1 | 288.8 | 1.44× |
| 2d_max_pooling | fp32 | 218.6 | 299.0 | 1.37× |
| 3d_conv | fp16 | 13870.0 | 35770.0 | 2.58× |
| 3d_conv | fp32 | 18190.0 | 33160.0 | 1.82× |
| argmax | fp16 | 31.2 | 30.9 | 0.99× |
| argmax | fp32 | 41.1 | 30.8 | 0.75× |
| batch_normalization | fp16 | 113.7 | 140.2 | 1.23× |
| batch_normalization | bf16 | 113.4 | 139.1 | 1.23× |
| batch_normalization | fp32 | 117.4 | 141.2 | 1.20× |
| batched_matmul | fp16 | 29.8 | 30.3 | 1.02× |
| batched_matmul | bf16 | 29.9 | 29.8 | 1.00× |
| batched_matmul | fp32 | 49.5 | 99.3 | 2.01× |
| bitonic_sort | fp16 | 8106.3 | 8105.4 | 1.00× |
| bitonic_sort | fp32 | 9271.4 | 9926.0 | 1.07× |
| block_sparse_attention | fp16 | 75.9 | 281.8 | 3.71× |
| cross_entropy | fp16 | 9.8 | 14.7 | 1.50× |
| cross_entropy | fp32 | 9.9 | 13.8 | 1.39× |
| dequantize_rowwise | fp32 | 18.4 | 21.5 | 1.17× |
| destindex | fp16 | 42.5 | 115.0 | 2.71× |
| destindex | bf16 | 42.4 | 115.3 | 2.72× |
| destindex | fp32 | 85.2 | 126.6 | 1.49× |
| destindex | int8 | 41.5 | 111.6 | 2.69× |
| dropout | fp16 | 18.8 | 18.9 | 1.01× |
| dropout | bf16 | 18.4 | 19.0 | 1.03× |
| dropout | fp32 | 34.2 | 35.7 | 1.04× |
| flash_attention | fp16 | 22740.0 | 17670.0 | 0.78× |
| flash_decode | fp32 | 47.4 | 169.2 | 3.57× |
| fused_activation | fp32 | 47.9 | 47.4 | 0.99× |
| gaussian_blur | fp16 | 1030.0 | 1500.0 | 1.46× |
| gaussian_blur | fp32 | 1270.0 | 1280.0 | 1.01× |
| histogramming | int32 | 793.7 | 1436.6 | 1.81× |
| interleave | fp16 | 21.4 | 22.4 | 1.04× |
| interleave | bf16 | 21.4 | 22.5 | 1.05× |
| interleave | fp32 | 43.9 | 51.4 | 1.17× |
| interleave | int8 | 13.2 | 16.9 | 1.28× |
| jacobi_stencil_2d | fp16 | 155.5 | 134.7 | 0.87× |
| jacobi_stencil_2d | bf16 | 155.3 | 134.5 | 0.87× |
| jacobi_stencil_2d | fp32 | 217.1 | 197.8 | 0.91× |
| kl_divergence | fp32 | 94.5 | 89.1 | 0.94× |
| l2_norm | fp16 | 14.6 | 19.2 | 1.32× |
| l2_norm | bf16 | 14.4 | 18.9 | 1.31× |
| l2_norm | fp32 | 27.5 | 30.4 | 1.11× |
| layernorm | fp16 | 19.8 | 32.9 | 1.66× |
| layernorm | bf16 | 20.9 | 33.4 | 1.60× |
| layernorm | fp32 | 33.0 | 33.3 | 1.01× |
| leaky_relu | fp16 | 26.3 | 27.2 | 1.03× |
| leaky_relu | bf16 | 27.1 | 28.0 | 1.03× |
| leaky_relu | fp32 | 55.8 | 58.4 | 1.05× |
| linear_self_attention | fp32 | 5107.7 | 20052.7 | 3.93× |
| matmul_fp32_fp16_fp8 | fp32 | 1340.0 | 871.7 | 0.65× |
| matmul_fp32_fp16_fp8 | fp16 | 518.0 | 467.2 | 0.90× |
| matmul_fp32_fp16_fp8 | fp8_e4m3fn | 257.1 | 220.8 | 0.86× |
| matmul_int8 | int8 | 338.0 | 341.1 | 1.01× |
| matrix_copy | fp16 | 17.8 | 18.1 | 1.02× |
| matrix_copy | bf16 | 18.0 | 18.0 | 1.00× |
| matrix_copy | fp32 | 31.2 | 28.0 | 0.90× |
| matrix_copy | int8 | 9.8 | 17.7 | 1.81× |
| matrix_transpose | fp16 | 50.5 | 50.8 | 1.01× |
| matrix_transpose | bf16 | 50.1 | 50.5 | 1.01× |
| matrix_transpose | fp32 | 99.5 | 101.7 | 1.02× |
| matrix_transpose | int8 | 28.7 | 34.0 | 1.18× |
| mean_reduction | fp16 | 16.8 | 19.4 | 1.16× |
| mean_reduction | bf16 | 16.8 | 26.8 | 1.59× |
| mean_reduction | fp32 | 28.4 | 32.5 | 1.15× |
| moe_topk_gating | fp16 | 14.7 | 29.1 | 1.98× |
| moe_topk_gating | bf16 | 14.6 | 29.1 | 1.99× |
| moe_topk_gating | fp32 | 14.7 | 29.2 | 1.99× |
| mul2 | fp16 | 14.8 | 15.2 | 1.02× |
| mul2 | bf16 | 13.4 | 15.3 | 1.14× |
| mul2 | fp32 | 22.1 | 22.1 | 1.00× |
| mul2 | int8 | 9.7 | 15.1 | 1.56× |
| quantize_global | fp32 | 17.7 | 18.3 | 1.03× |
| radix_sort | int32 | 2379.0 | 2985.8 | 1.26× |
| relu | fp16 | 14.9 | 13.0 | 0.87× |
| relu | bf16 | 14.8 | 13.0 | 0.87× |
| relu | fp32 | 22.3 | 22.4 | 1.01× |
| relu | int8 | 14.8 | 17.0 | 1.14× |
| reverse_array | fp16 | 17.0 | 14.4 | 0.84× |
| reverse_array | bf16 | 16.7 | 14.7 | 0.88× |
| reverse_array | fp32 | 21.5 | 24.5 | 1.14× |
| reverse_array | int8 | 14.4 | 13.6 | 0.94× |
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
| swiglu | fp16 | 69.0 | 99.8 | 1.45× |
| swiglu | bf16 | 72.9 | 100.8 | 1.38× |
| swiglu | fp32 | 137.9 | 148.3 | 1.08× |
| top_k_selection | fp32 | 200.6 | 140.2 | 0.70× |
| vector_add | fp16 | 18.3 | 18.4 | 1.01× |
| vector_add | bf16 | 18.1 | 18.6 | 1.03× |
| vector_add | fp32 | 34.3 | 34.4 | 1.00× |
| vector_add | int8 | 12.0 | 17.2 | 1.43× |
| weight_dequant | fp16 | 62.7 | 188.6 | 3.01× |
| weight_dequant | bf16 | 63.3 | 188.6 | 2.98× |
| weight_dequant | fp32 | 117.0 | 224.4 | 1.92× |

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
