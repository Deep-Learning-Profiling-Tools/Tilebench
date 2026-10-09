#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void relu_kernel_kernel(float* __restrict__ output, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) relu_kernel_kernel(float* __restrict__ output, const float* __restrict__ x) {
  float x_reg[8];
  float output_reg[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(x_reg + (i * 4)) = *(float4*)(x + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float value = x_reg[i_1];
    output_reg[i_1] = ((0x0p+0f/*0.000000e+00*/ <= value) ? value : 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(float4*)(output + (((((int)blockIdx.x) * 2048) + (i_2 * 1024)) + (((int)threadIdx.x) * 4))) = *(float4*)(output_reg + (i_2 * 4));
  }
}

