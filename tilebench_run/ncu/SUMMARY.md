# NCU Sweep Summary — all operators

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** autotune-winner cfg at sweep-max input case, `--set full --import-source on`, `--launch-skip 3 --launch-count 1`

Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the `<backend>_<dtype>.ncu-rep` files in that directory.

## Headline duration table (µs)

| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |
|---|---|---:|---:|---:|
| 1d_conv | fp16 | 404.4 | 583.2 | 1.44× |
| 1d_conv | fp32 | 588.9 | 452.9 | 0.77× |
| 2d_conv | fp16 | 556.1 | 797.6 | 1.43× |
| 2d_conv | fp32 | 1860.0 | 5390.0 | 2.90× |
| 2d_max_pooling | fp16 | 229.2 | 598.4 | 2.61× |
| 2d_max_pooling | bf16 | 228.9 | 578.6 | 2.53× |
| 2d_max_pooling | fp32 | 265.5 | 592.6 | 2.23× |
| 3d_conv | fp16 | 264.0 | 251.3 | 0.95× |
| 3d_conv | fp32 | 235.8 | 216.7 | 0.92× |
| argmax | fp16 | 30.1 | 47.8 | 1.59× |
| argmax | fp32 | 40.7 | 39.4 | 0.97× |
| batch_normalization | fp16 | 113.6 | 140.1 | 1.23× |
| batch_normalization | bf16 | 113.3 | 139.6 | 1.23× |
| batch_normalization | fp32 | 117.2 | 141.5 | 1.21× |
| batched_matmul | fp16 | 57.7 | 60.5 | 1.05× |
| batched_matmul | bf16 | 57.4 | 60.2 | 1.05× |
| batched_matmul | fp32 | 293.7 | 176.3 | 0.60× |
| bitonic_sort | fp16 | 8082.0 | 8587.4 | 1.06× |
| bitonic_sort | fp32 | 9273.6 | 9665.9 | 1.04× |
| block_sparse_attention | fp16 | 75.9 | 281.8 | 3.71× |
| cross_entropy | fp16 | 9.8 | 14.5 | 1.48× |
| cross_entropy | fp32 | 9.9 | 13.5 | 1.37× |
| dequantize_rowwise | fp32 | 19.2 | 21.2 | 1.10× |
| destindex | fp16 | 281.9 | 282.1 | 1.00× |
| destindex | bf16 | 281.9 | 282.3 | 1.00× |
| destindex | fp32 | 282.6 | 282.8 | 1.00× |
| destindex | int8 | 282.2 | 282.0 | 1.00× |
| dropout | fp16 | 18.4 | 18.8 | 1.02× |
| dropout | bf16 | 18.5 | 18.9 | 1.02× |
| dropout | fp32 | 34.6 | 36.1 | 1.04× |
| flash_attention | fp16 | 22770.0 | 17670.0 | 0.78× |
| flash_decode | fp32 | 47.4 | 168.9 | 3.56× |
| fused_activation | fp32 | 47.3 | 48.0 | 1.01× |
| gaussian_blur | fp16 | 1790.0 | 4600.0 | 2.57× |
| gaussian_blur | fp32 | 1610.0 | 5310.0 | 3.30× |
| histogramming | int32 | 1853.4 | 1972.4 | 1.06× |
| interleave | fp16 | 21.4 | 22.4 | 1.04× |
| interleave | bf16 | 21.4 | 22.5 | 1.05× |
| interleave | fp32 | 43.9 | 51.4 | 1.17× |
| interleave | int8 | 13.2 | 16.9 | 1.28× |
| jacobi_stencil_2d | fp16 | 155.3 | 135.0 | 0.87× |
| jacobi_stencil_2d | bf16 | 155.4 | 134.5 | 0.87× |
| jacobi_stencil_2d | fp32 | 217.4 | 198.3 | 0.91× |
| kl_divergence | fp32 | 95.0 | 88.8 | 0.93× |
| l2_norm | fp16 | 16.6 | 19.8 | 1.19× |
| l2_norm | bf16 | 16.7 | 19.0 | 1.14× |
| l2_norm | fp32 | 34.6 | 42.9 | 1.24× |
| layernorm | fp16 | — | 33.2 | — |
| layernorm | bf16 | 21.0 | 33.2 | 1.58× |
| layernorm | fp32 | 33.1 | 33.6 | 1.01× |
| leaky_relu | fp16 | 26.3 | 28.0 | 1.06× |
| leaky_relu | bf16 | 26.9 | 27.5 | 1.02× |
| leaky_relu | fp32 | 56.2 | 57.2 | 1.02× |
| linear_self_attention | fp32 | 4900.0 | 19570.0 | 3.99× |
| matmul_fp32_fp16_fp8 | fp32 | 6130.0 | 981.2 | 0.16× |
| matmul_fp32_fp16_fp8 | fp16 | 671.8 | 516.1 | 0.77× |
| matmul_fp32_fp16_fp8 | fp8_e4m3fn | 575.5 | 230.8 | 0.40× |
| matmul_fp32_fp16_fp8 | fp8_e5m2 | 580.1 | 245.9 | 0.42× |
| matmul_int8 | int8 | 475.5 | 345.6 | 0.73× |
| matrix_copy | fp16 | 17.8 | 18.1 | 1.02× |
| matrix_copy | bf16 | 18.0 | 18.0 | 1.00× |
| matrix_copy | fp32 | 31.2 | 28.0 | 0.90× |
| matrix_copy | int8 | 9.8 | 17.7 | 1.81× |
| matrix_transpose | fp16 | 50.2 | 50.0 | 1.00× |
| matrix_transpose | bf16 | 50.9 | 50.6 | 0.99× |
| matrix_transpose | fp32 | 98.7 | 102.3 | 1.04× |
| matrix_transpose | int8 | 29.1 | 34.3 | 1.18× |
| mean_reduction | fp16 | 16.9 | 58.8 | 3.49× |
| mean_reduction | bf16 | 19.0 | 20.7 | 1.09× |
| mean_reduction | fp32 | 30.0 | 30.1 | 1.00× |
| moe_topk_gating | fp16 | 25.3 | 29.6 | 1.17× |
| moe_topk_gating | bf16 | 24.9 | 29.5 | 1.18× |
| moe_topk_gating | fp32 | 25.2 | 29.4 | 1.17× |
| mul2 | fp16 | 14.8 | 15.2 | 1.03× |
| mul2 | bf16 | 13.6 | 15.0 | 1.11× |
| mul2 | fp32 | 22.0 | 22.3 | 1.01× |
| mul2 | int8 | 9.9 | 15.0 | 1.51× |
| quantize_global | fp32 | 17.7 | 18.5 | 1.04× |
| radix_sort | int32 | 2390.2 | 187.8 | 0.08× |
| relu | fp16 | 14.9 | 13.5 | 0.91× |
| relu | bf16 | 14.8 | 13.2 | 0.89× |
| relu | fp32 | 22.2 | 22.7 | 1.02× |
| relu | int8 | 15.1 | 17.2 | 1.14× |
| reverse_array | fp16 | 16.5 | 14.1 | 0.86× |
| reverse_array | bf16 | 16.6 | 14.3 | 0.86× |
| reverse_array | fp32 | 21.5 | 25.0 | 1.16× |
| reverse_array | int8 | 14.6 | 13.8 | 0.95× |
| rmsnorm | fp16 | 17.9 | 20.6 | 1.15× |
| rmsnorm | bf16 | 17.9 | 21.3 | 1.19× |
| rmsnorm | fp32 | 30.8 | 36.9 | 1.20× |
| rope | fp16 | 90.8 | 201.7 | 2.22× |
| rope | fp32 | 131.8 | 212.4 | 1.61× |
| sigmoid | fp16 | 30.4 | 51.0 | 1.68× |
| sigmoid | bf16 | 30.2 | 50.9 | 1.68× |
| sigmoid | fp32 | 55.5 | 64.9 | 1.17× |
| softmax | fp16 | 23.8 | 43.2 | 1.82× |
| softmax | fp32 | 32.6 | 44.5 | 1.36× |
| streamk_matmul | fp16 | 2646.0 | 14306.1 | 5.41× |
| streamk_matmul | bf16 | 2595.7 | 14439.4 | 5.56× |
| streamk_matmul | fp32 | 17363.7 | 16225.6 | 0.93× |
| swiglu | fp16 | 69.1 | 99.5 | 1.44× |
| swiglu | bf16 | 72.5 | 100.4 | 1.38× |
| swiglu | fp32 | 136.9 | 147.1 | 1.07× |
| top_k_selection | fp32 | 1153.2 | 1344.5 | 1.17× |
| vector_add | fp16 | — | 18.5 | — |
| vector_add | bf16 | — | 19.2 | — |
| vector_add | fp32 | — | 34.6 | — |
| vector_add | int8 | — | 17.1 | — |
| weight_dequant | fp16 | 69.0 | 207.8 | 3.01× |
| weight_dequant | bf16 | 68.8 | 206.3 | 3.00× |
| weight_dequant | fp32 | 118.8 | 201.6 | 1.70× |

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
