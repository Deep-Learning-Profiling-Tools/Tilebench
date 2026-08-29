# NCU Sweep Summary — all operators

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** autotune-winner cfg at sweep-max input case, `--set full --import-source on`, `--launch-skip 3 --launch-count 1`

Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the `<backend>_<dtype>.ncu-rep` files in that directory.

## Headline duration table (µs)

| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |
|---|---|---:|---:|---:|
| leaky_relu | fp16 | 33.5 | 34.3 | 1.03× |
| leaky_relu | bf16 | 34.6 | 35.0 | 1.01× |
| leaky_relu | fp32 | 62.2 | 64.2 | 1.03× |
| dropout | fp16 | 23.7 | 24.3 | 1.03× |
| dropout | bf16 | 23.5 | 24.0 | 1.02× |
| dropout | fp32 | 42.8 | 42.9 | 1.00× |
| 2d_max_pooling | fp16 | 203.4 | 290.7 | 1.43× |
| 2d_max_pooling | bf16 | 205.1 | 290.7 | 1.42× |
| 2d_max_pooling | fp32 | 227.5 | 306.4 | 1.35× |
| 3d_conv | fp16 | 13850.0 | 35670.0 | 2.58× |
| 3d_conv | fp32 | 18200.0 | 33080.0 | 1.82× |
| argmax | fp16 | 32.5 | 33.2 | 1.02× |
| argmax | fp32 | 44.0 | 38.8 | 0.88× |
| bitonic_sort | fp16 | 8116.7 | 8107.6 | 1.00× |
| bitonic_sort | fp32 | 9277.0 | 9114.4 | 0.98× |
| cross_entropy | fp16 | 10.7 | 14.1 | 1.32× |
| cross_entropy | fp32 | 10.2 | 14.0 | 1.37× |
| dropout | fp16 | 18.2 | 19.1 | 1.05× |
| dropout | bf16 | 18.5 | 18.9 | 1.02× |
| dropout | fp32 | 34.4 | 34.5 | 1.00× |
| cross_entropy | fp16 | 9.9 | 15.0 | 1.53× |
| cross_entropy | fp32 | 10.5 | 13.9 | 1.33× |
| gaussian_blur | fp16 | 1030.0 | 1510.0 | 1.47× |
| gaussian_blur | fp32 | 1280.0 | 1290.0 | 1.01× |
| jacobi_stencil_2d | fp16 | 163.2 | 136.3 | 0.84× |
| jacobi_stencil_2d | bf16 | 163.7 | 135.9 | 0.83× |
| jacobi_stencil_2d | fp32 | 222.5 | 185.2 | 0.83× |
| layernorm | fp16 | 24.5 | 38.0 | 1.55× |
| layernorm | bf16 | 24.6 | 39.1 | 1.59× |
| layernorm | fp32 | 41.3 | 38.9 | 0.94× |
| matrix_copy | fp16 | 21.5 | 22.4 | 1.04× |
| matrix_copy | bf16 | 22.0 | 21.8 | 0.99× |
| matrix_copy | fp32 | 37.1 | 36.0 | 0.97× |
| matrix_copy | int8 | 13.4 | 13.9 | 1.03× |
| matrix_transpose | fp16 | 56.4 | 56.8 | 1.01× |
| matrix_transpose | bf16 | 57.1 | 55.9 | 0.98× |
| matrix_transpose | fp32 | 106.6 | 107.1 | 1.01× |
| matrix_transpose | int8 | 32.7 | 32.5 | 0.99× |
| mean_reduction | fp16 | 21.1 | 22.0 | 1.04× |
| mean_reduction | bf16 | 20.0 | 25.1 | 1.26× |
| mean_reduction | fp32 | 37.2 | 39.0 | 1.05× |
| moe_topk_gating | fp16 | 15.7 | 30.7 | 1.96× |
| moe_topk_gating | bf16 | 16.1 | 30.4 | 1.89× |
| moe_topk_gating | fp32 | 15.7 | 30.7 | 1.96× |
| mul2 | fp16 | 18.1 | 19.6 | 1.09× |
| mul2 | bf16 | 17.9 | 18.9 | 1.06× |
| mul2 | fp32 | 29.5 | 30.3 | 1.02× |
| mul2 | int8 | 11.3 | 12.9 | 1.15× |
| relu | fp16 | 17.2 | 17.4 | 1.01× |
| relu | bf16 | 17.3 | 17.9 | 1.04× |
| relu | fp32 | 29.9 | 30.6 | 1.02× |
| relu | int8 | 11.9 | 15.2 | 1.28× |
| reverse_array | fp16 | 19.6 | 20.3 | 1.04× |
| reverse_array | bf16 | 19.6 | 20.9 | 1.07× |
| reverse_array | fp32 | 28.4 | 29.9 | 1.05× |
| reverse_array | int8 | 15.1 | 15.1 | 1.00× |
| swiglu | fp16 | 77.4 | 103.5 | 1.34× |
| swiglu | bf16 | 78.9 | 104.2 | 1.32× |
| swiglu | fp32 | 145.9 | 149.7 | 1.03× |
| vector_add | fp16 | 24.0 | 23.9 | 1.00× |
| vector_add | bf16 | 23.7 | 24.7 | 1.04× |
| vector_add | fp32 | 41.5 | 41.6 | 1.00× |
| vector_add | int8 | 14.8 | 16.0 | 1.08× |
| weight_dequant | fp16 | 69.7 | 188.6 | 2.71× |
| weight_dequant | bf16 | 70.2 | 188.6 | 2.69× |
| weight_dequant | fp32 | 126.9 | 196.2 | 1.55× |
| 1d_conv | fp16 | 2520.0 | 4510.0 | 1.79× |
| 1d_conv | fp32 | 3080.0 | 5320.0 | 1.73× |
| 2d_conv | fp16 | 552.0 | 797.2 | 1.44× |
| 2d_conv | fp32 | 626.5 | 934.1 | 1.49× |
| batch_normalization | fp16 | 35.2 | 51.1 | 1.45× |
| batch_normalization | bf16 | 34.9 | 50.6 | 1.45× |
| batch_normalization | fp32 | 66.2 | 58.0 | 0.88× |
| batched_matmul | fp16 | 32.1 | 34.9 | 1.09× |
| batched_matmul | bf16 | 32.1 | 34.8 | 1.08× |
| batched_matmul | fp32 | 57.4 | 99.8 | 1.74× |
| block_sparse_attention | fp16 | 75.7 | 223.2 | 2.95× |
| dequantize_rowwise | int8 | 22.2 | 26.4 | 1.19× |
| destindex | fp16 | 49.4 | 117.2 | 2.37× |
| destindex | bf16 | 49.1 | 116.9 | 2.38× |
| destindex | fp32 | 93.3 | 128.5 | 1.38× |
| destindex | int8 | 42.8 | 113.0 | 2.64× |
| flash_attention | fp16 | 19340.0 | 17680.0 | 0.91× |
| flash_decode | fp32 | 47.4 | 168.3 | 3.55× |
| fused_activation | fp32 | 53.8 | 54.6 | 1.02× |
| histogramming | int32 | 822.2 | 1427.3 | 1.74× |
| interleave | fp16 | 28.4 | 29.5 | 1.04× |
| interleave | bf16 | 28.5 | 29.1 | 1.02× |
| interleave | fp32 | 51.7 | 51.8 | 1.00× |
| interleave | int8 | 17.0 | 17.4 | 1.03× |
| kl_divergence | fp32 | 106.0 | 99.2 | 0.94× |
| l2_norm | fp16 | 18.0 | 23.0 | 1.28× |
| l2_norm | bf16 | 17.8 | 23.2 | 1.30× |
| l2_norm | fp32 | 32.5 | 33.7 | 1.04× |
| linear_self_attention | fp32 | 203.3 | 499.0 | 2.45× |
| matmul_fp32_fp16_fp8 | fp32 | 1340.0 | 871.7 | 0.65× |
| matmul_fp32_fp16_fp8 | fp16 | 518.0 | 467.2 | 0.90× |
| matmul_fp32_fp16_fp8 | fp8_e4m3fn | 257.1 | 220.8 | 0.86× |
| matmul_int8 | int8 | 338.0 | 341.1 | 1.01× |
| quantize_global | fp32 | 23.5 | 23.3 | 0.99× |
| radix_sort | int32 | 1331.2 | 2529.9 | 1.90× |
| rmsnorm | fp16 | 21.8 | 23.8 | 1.09× |
| rmsnorm | bf16 | 21.7 | 23.1 | 1.06× |
| rmsnorm | fp32 | 35.6 | 36.1 | 1.01× |
| rope | fp16 | 54.5 | 55.1 | 1.01× |
| rope | fp32 | 106.7 | 106.4 | 1.00× |
| sigmoid | fp16 | 34.3 | 52.0 | 1.51× |
| sigmoid | bf16 | 34.2 | 51.4 | 1.50× |
| sigmoid | fp32 | 63.1 | 65.1 | 1.03× |
| softmax | fp16 | 25.4 | 44.2 | 1.74× |
| softmax | fp32 | 37.9 | 47.4 | 1.25× |
| streamk_matmul | fp16 | 1962.6 | 2220.7 | 1.13× |
| streamk_matmul | bf16 | 2073.8 | 2171.1 | 1.05× |
| streamk_matmul | fp32 | 3628.5 | 3733.4 | 1.03× |
| top_k_selection | fp32 | 198.0 | 135.4 | 0.68× |

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