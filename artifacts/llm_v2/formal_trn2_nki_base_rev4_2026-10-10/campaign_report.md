| task | model | status | rounds | valid | attempts | repairs | best geomean ms (round) | valid cases | tokens (in/out/reasoning) | exact | reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nki/1d_conv/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.9774 (4) | 20 | 222551/358235/None | True | - |
| nki/2d_conv/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0855 (5) | 20 | 230673/434064/None | True | - |
| nki/3d_conv/fp16 | claude | complete | 5 | 2 | 5 | 0 | 1.3165 (4) | 20 | 235527/498013/None | True | - |
| nki/batch_normalization/fp16 | claude | complete | 5 | 1 | 5 | 0 | 0.1687 (5) | 20 | 215787/521672/None | True | - |
| nki/batched_matmul/fp16 | claude | complete | 5 | 4 | 5 | 0 | 0.0616 (5) | 20 | 248206/473800/None | True | - |
| nki/bitonic_sort/fp16 | claude | complete | 5 | 1 | 5 | 0 | 7.9442 (5) | 20 | 217916/401220/None | True | - |
| nki/cross_entropy/fp16 | claude | complete | 5 | 3 | 5 | 0 | 0.0554 (2) | 20 | 214071/335704/None | True | - |
| nki/dequantize_rowwise/int8 | claude | complete | 5 | 5 | 5 | 0 | 0.0355 (4) | 20 | 207069/179521/None | True | - |
| nki/destindex/fp16 | claude | complete | 5 | 3 | 5 | 0 | 0.2823 (3) | 20 | 220434/193467/None | True | - |
| nki/dropout/fp16 | claude | complete | 5 | 4 | 5 | 0 | 0.0978 (3) | 20 | 210393/165760/None | True | r4:violation |
| nki/flash_decode/fp32 | claude | complete | 5 | 0 | 5 | 0 | - | - | 206575/372302/None | True | r3:compliant, r4:compliant, r5:compliant |
| nki/fused_activation/fp32 | claude | complete | 5 | 5 | 5 | 0 | 0.2269 (1) | 20 | 208983/164553/None | True | - |
| nki/histogramming/int32 | claude | complete | 5 | 2 | 5 | 0 | 2.2536 (2) | 20 | 211129/574992/None | True | - |
| nki/interleave/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.1199 (3) | 20 | 216060/250905/None | True | - |
| nki/jacobi_stencil_2d/fp16 | claude | complete | 5 | 4 | 5 | 0 | 0.1783 (4) | 20 | 227887/353527/None | True | r3:compliant, r4:compliant, r5:compliant |
| nki/kl_divergence/fp32 | claude | complete | 5 | 5 | 5 | 0 | 0.1119 (1) | 20 | 213414/466052/None | True | - |
| nki/l2_norm/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0739 (5) | 20 | 209932/385525/None | True | - |
| nki/layernorm/fp16 | claude | complete | 5 | 3 | 5 | 0 | 0.1313 (4) | 20 | 229219/593929/None | True | - |
| nki/leaky_relu/fp16 | claude | complete | 5 | 4 | 5 | 0 | 0.1460 (1) | 20 | 203348/270090/None | True | - |
| nki/linear_self_attention/fp32 | claude | complete | 5 | 1 | 5 | 0 | 0.0609 (5) | 20 | 219599/350044/None | True | - |
| nki/matmul_fp32_fp16_fp8/fp16 | claude | complete | 5 | 5 | 5 | 0 | 1.9961 (5) | 20 | 212404/316012/None | True | - |
| nki/matrix_copy/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0522 (5) | 20 | 203129/120837/None | True | - |
| nki/matrix_transpose/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.2408 (4) | 20 | 206663/282342/None | True | - |
| nki/mean_reduction/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0745 (2) | 20 | 231381/457982/None | True | - |
| nki/moe_topk_gating/fp16 | claude | complete | 5 | 0 | 5 | 0 | - | - | 214783/319394/None | True | - |
| nki/mul2/fp16 | claude | complete | 5 | 3 | 5 | 0 | 0.0676 (5) | 20 | 192909/106867/None | True | - |
| nki/quantize_global/fp32 | claude | complete | 5 | 5 | 5 | 0 | 0.0963 (2) | 20 | 202729/168477/None | True | - |
| nki/relu/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0678 (5) | 20 | 205111/189995/None | True | - |
| nki/rmsnorm/fp16 | claude | complete | 5 | 3 | 5 | 0 | 0.0774 (4) | 20 | 224742/520751/None | True | - |
| nki/rope/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.3394 (5) | 20 | 221151/389702/None | True | - |
| nki/sigmoid/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.1420 (5) | 20 | 217061/278137/None | True | - |
| nki/softmax/fp16 | claude | complete | 5 | 4 | 5 | 0 | 0.1318 (4) | 20 | 219944/394753/None | True | - |
| nki/streamk_matmul/fp16 | claude | complete | 5 | 0 | 5 | 0 | - | - | 218651/507833/None | True | - |
| nki/swiglu/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.3557 (5) | 20 | 207622/184802/None | True | - |
| nki/vector_add/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.0949 (5) | 20 | 208796/184518/None | True | - |
| nki/weight_dequant/fp16 | claude | complete | 5 | 5 | 5 | 0 | 0.1462 (5) | 20 | 209481/355819/None | True | - |
| nki/1d_conv/fp16 | gpt | complete | 5 | 2 | 5 | 0 | 1.3903 (4) | 20 | 135299/47009/37650 | True | - |
| nki/2d_conv/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 0.2607 (3) | 20 | 140931/79289/67921 | True | - |
| nki/3d_conv/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 2.4006 (4) | 20 | 138908/81342/69218 | True | - |
| nki/batch_normalization/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 0.2543 (2) | 20 | 146210/61009/46616 | True | - |
| nki/batched_matmul/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0827 (2) | 20 | 140232/73031/63164 | True | r1:compliant, r2:compliant, r3:compliant, r4:compliant, r5:compliant |
| nki/bitonic_sort/fp16 | gpt | complete | 5 | 1 | 5 | 0 | 44.7082 (5) | 20 | 138697/65199/51348 | True | - |
| nki/cross_entropy/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 0.0869 (5) | 20 | 131063/33436/28011 | True | - |
| nki/dequantize_rowwise/int8 | gpt | complete | 5 | 5 | 5 | 0 | 0.0364 (5) | 20 | 127203/36263/32200 | True | - |
| nki/destindex/fp16 | gpt | complete | 5 | 3 | 5 | 0 | 0.3525 (1) | 20 | 135590/24730/20249 | True | - |
| nki/dropout/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0996 (3) | 20 | 128062/27259/23529 | True | - |
| nki/flash_decode/fp32 | gpt | complete | 5 | 0 | 5 | 0 | - | - | 133536/52335/42232 | True | r1:compliant, r2:compliant, r3:compliant, r4:compliant, r5:compliant |
| nki/fused_activation/fp32 | gpt | complete | 5 | 5 | 5 | 0 | 0.2235 (5) | 20 | 128584/35097/30389 | True | - |
| nki/histogramming/int32 | gpt | complete | 5 | 3 | 5 | 0 | 90.1709 (5) | 20 | 141779/65316/49820 | True | - |
| nki/interleave/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.1214 (2) | 20 | 126782/40767/36712 | True | - |
| nki/jacobi_stencil_2d/fp16 | gpt | complete | 5 | 3 | 5 | 0 | 0.2827 (3) | 20 | 134218/69841/58201 | True | - |
| nki/kl_divergence/fp32 | gpt | complete | 5 | 2 | 5 | 0 | 0.1837 (3) | 20 | 131123/52047/45545 | True | - |
| nki/l2_norm/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0776 (4) | 20 | 127427/35677/30427 | True | - |
| nki/layernorm/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.1602 (2) | 20 | 134514/79562/71393 | True | - |
| nki/leaky_relu/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.1430 (4) | 20 | 126792/32777/28290 | True | - |
| nki/linear_self_attention/fp32 | gpt | complete | 5 | 2 | 5 | 0 | 0.0492 (4) | 20 | 138539/51900/40691 | True | - |
| nki/matmul_fp32_fp16_fp8/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 2.4668 (5) | 20 | 133768/82296/72974 | True | - |
| nki/matrix_copy/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0513 (5) | 20 | 125875/29578/26920 | True | - |
| nki/matrix_transpose/fp16 | gpt | complete | 5 | 4 | 5 | 0 | 0.2846 (4) | 20 | 128520/51991/47582 | True | - |
| nki/mean_reduction/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0890 (4) | 20 | 129253/52566/47607 | True | - |
| nki/moe_topk_gating/fp16 | gpt | complete | 5 | 2 | 5 | 0 | 0.1813 (3) | 20 | 135806/39689/31505 | True | - |
| nki/mul2/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0673 (5) | 20 | 128426/28457/25152 | True | - |
| nki/quantize_global/fp32 | gpt | complete | 5 | 5 | 5 | 0 | 0.0958 (5) | 20 | 126239/27261/24481 | True | - |
| nki/relu/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0674 (1) | 20 | 127909/26655/23500 | True | - |
| nki/rmsnorm/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0972 (4) | 20 | 130088/41827/36420 | True | - |
| nki/rope/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.2791 (5) | 20 | 130133/57944/52386 | True | - |
| nki/sigmoid/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.1498 (5) | 20 | 131054/48156/42171 | True | - |
| nki/softmax/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0996 (5) | 20 | 129608/46251/40618 | True | - |
| nki/streamk_matmul/fp16 | gpt | complete | 5 | 1 | 5 | 0 | 4.1776 (5) | 20 | 143136/56707/43110 | True | - |
| nki/swiglu/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.3391 (4) | 20 | 127928/26130/22055 | True | - |
| nki/vector_add/fp16 | gpt | complete | 5 | 5 | 5 | 0 | 0.0963 (5) | 20 | 126621/24500/21737 | True | - |
| nki/weight_dequant/fp16 | gpt | complete | 5 | 3 | 5 | 0 | 0.1664 (1) | 20 | 132197/45213/40647 | True | r2:compliant, r5:compliant |
