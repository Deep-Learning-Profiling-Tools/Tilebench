# NCU Sweep Summary — all operators

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** autotune-winner cfg at sweep-max input case, `--set full --import-source on`, `--launch-skip 3 --launch-count 1`

Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the `<backend>_<dtype>.ncu-rep` files in that directory.

## Headline duration table (µs)

| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |
|---|---|---:|---:|---:|
| 1d_conv | fp16 | 531.2 | 597.8 | 1.13× |
| 1d_conv | fp32 | 627.8 | 451.2 | 0.72× |
| 2d_conv | fp16 | 556.1 | 797.6 | 1.43× |
| 2d_conv | fp32 | 1860.0 | 5390.0 | 2.90× |
| 2d_max_pooling | fp16 | 229.2 | 598.4 | 2.61× |
| 2d_max_pooling | bf16 | 228.9 | 578.6 | 2.53× |
| 2d_max_pooling | fp32 | 265.5 | 592.6 | 2.23× |
| 3d_conv | fp16 | 264.0 | 251.2 | 0.95× |
| 3d_conv | fp32 | 235.8 | 216.5 | 0.92× |
| argmax | fp16 | 30.1 | 47.8 | 1.59× |
| argmax | fp32 | 40.7 | 39.4 | 0.97× |
| batch_normalization | fp16 | 113.6 | 140.1 | 1.23× |
| batch_normalization | bf16 | 113.3 | 139.6 | 1.23× |
| batch_normalization | fp32 | 117.2 | 141.5 | 1.21× |
| batched_matmul | fp16 | 29.7 | 30.1 | 1.01× |
| batched_matmul | bf16 | 28.9 | 29.8 | 1.03× |
| batched_matmul | fp32 | 118.2 | 18980.0 | 160.60× |
| bitonic_sort | fp16 | 8082.0 | 8587.4 | 1.06× |
| bitonic_sort | fp32 | 9273.6 | 9665.9 | 1.04× |
| block_sparse_attention | fp16 | 76.7 | 280.5 | 3.65× |
| cross_entropy | fp16 | 9.8 | 14.5 | 1.48× |
| cross_entropy | fp32 | 9.9 | 13.5 | 1.37× |
| dequantize_rowwise | fp32 | 19.2 | 21.2 | 1.10× |
| destindex | fp16 | 16.6 | 16.3 | 0.98× |
| destindex | bf16 | 16.6 | 16.4 | 0.99× |
| destindex | fp32 | 16.8 | 16.7 | 1.00× |
| destindex | int8 | 284.2 | 283.8 | 1.00× |
| dropout | fp16 | 18.4 | 18.8 | 1.02× |
| dropout | bf16 | 18.5 | 18.9 | 1.02× |
| dropout | fp32 | 34.6 | 36.1 | 1.04× |
| flash_attention | fp16 | 22770.0 | 17700.0 | 0.78× |
| flash_decode | fp32 | 47.4 | 168.3 | 3.55× |
| fused_activation | fp32 | 47.3 | 48.0 | 1.01× |
| gaussian_blur | fp16 | 1790.0 | 4600.0 | 2.57× |
| gaussian_blur | fp32 | 1610.0 | 5310.0 | 3.30× |
| histogramming | int32 | 1853.4 | 1972.4 | 1.06× |
| interleave | fp16 | 21.4 | 22.3 | 1.04× |
| interleave | bf16 | 21.4 | 22.3 | 1.04× |
| interleave | fp32 | 43.9 | 51.0 | 1.16× |
| interleave | int8 | 13.2 | 16.5 | 1.25× |
| jacobi_stencil_2d | fp16 | 155.3 | 135.0 | 0.87× |
| jacobi_stencil_2d | bf16 | 155.4 | 134.5 | 0.87× |
| jacobi_stencil_2d | fp32 | 217.4 | 198.3 | 0.91× |
| kl_divergence | fp32 | 136.3 | 136.9 | 1.00× |
| l2_norm | fp16 | 17.2 | 18.5 | 1.07× |
| l2_norm | bf16 | 17.2 | 18.1 | 1.05× |
| l2_norm | fp32 | 30.3 | 42.5 | 1.40× |
| layernorm | fp16 | 19.7 | 33.2 | 1.69× |
| layernorm | bf16 | 21.0 | 33.2 | 1.58× |
| layernorm | fp32 | 33.1 | 33.6 | 1.01× |
| leaky_relu | fp16 | 26.3 | 28.0 | 1.06× |
| leaky_relu | bf16 | 26.9 | 27.5 | 1.02× |
| leaky_relu | fp32 | 56.2 | 57.2 | 1.02× |
| linear_self_attention | fp32 | 4890.0 | 19570.0 | 4.00× |
| matmul_fp32_fp16_fp8 | fp32 | 6010.0 | 872.0 | 0.15× |
| matmul_fp32_fp16_fp8 | fp16 | 561.8 | 467.2 | 0.83× |
| matmul_fp32_fp16_fp8 | fp8_e4m3fn | 276.6 | 223.2 | 0.81× |
| matmul_fp32_fp16_fp8 | fp8_e5m2 | 283.9 | 234.1 | 0.82× |
| matmul_int8 | int8 | 475.5 | 342.4 | 0.72× |
| matrix_copy | fp16 | 17.8 | 18.1 | 1.02× |
| matrix_copy | bf16 | 18.0 | 18.0 | 1.00× |
| matrix_copy | fp32 | 31.2 | 28.0 | 0.90× |
| matrix_copy | int8 | 9.8 | 17.7 | 1.81× |
| matrix_transpose | fp16 | 50.2 | 50.0 | 1.00× |
| matrix_transpose | bf16 | 50.9 | 50.6 | 0.99× |
| matrix_transpose | fp32 | 98.7 | 102.3 | 1.04× |
| matrix_transpose | int8 | 29.1 | 34.3 | 1.18× |
| mean_reduction | fp16 | 17.1 | 19.6 | 1.15× |
| mean_reduction | bf16 | 17.2 | 26.6 | 1.55× |
| mean_reduction | fp32 | 29.9 | 32.7 | 1.09× |
| moe_topk_gating | fp16 | 14.8 | 29.5 | 1.99× |
| moe_topk_gating | bf16 | 14.7 | 30.3 | 2.07× |
| moe_topk_gating | fp32 | 14.7 | 29.4 | 2.00× |
| mul2 | fp16 | 14.8 | 15.3 | 1.03× |
| mul2 | bf16 | 13.6 | 15.3 | 1.13× |
| mul2 | fp32 | 22.0 | 22.4 | 1.02× |
| mul2 | int8 | 9.9 | 15.3 | 1.55× |
| quantize_global | fp32 | 17.7 | 18.5 | 1.04× |
| radix_sort | int32 | 2390.2 | 2989.9 | 1.25× |
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
| streamk_matmul | fp16 | 3482.7 | 14664.0 | 4.21× |
| streamk_matmul | bf16 | 3413.1 | 14662.2 | 4.30× |
| streamk_matmul | fp32 | 33424.7 | 16331.8 | 0.49× |
| swiglu | fp16 | 69.1 | 99.5 | 1.44× |
| swiglu | bf16 | 72.5 | 100.4 | 1.38× |
| swiglu | fp32 | 136.9 | 147.1 | 1.07× |
| top_k_selection | fp32 | 1153.2 | 1344.5 | 1.17× |
| vector_add | fp16 | 18.1 | 18.6 | 1.03× |
| vector_add | bf16 | 17.9 | 18.5 | 1.04× |
| vector_add | fp32 | 33.8 | 34.9 | 1.03× |
| vector_add | int8 | 12.2 | 17.0 | 1.39× |
| weight_dequant | fp16 | 62.0 | 326.8 | 5.27× |
| weight_dequant | bf16 | 62.3 | 328.7 | 5.28× |
| weight_dequant | fp32 | 117.1 | 326.0 | 2.79× |

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
