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

extern "C" __global__ void add_kernel_kernel(float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y);
extern "C" __global__ void __launch_bounds__(64, 1) add_kernel_kernel(float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y) {
  float x_reg[16];
  float y_reg[16];
  float out_reg[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(float4*)(x_reg + (i * 4)) = *(float4*)(x + (((((int)blockIdx.x) * 1024) + (i * 256)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float4*)(y_reg + (i_1 * 4)) = *(float4*)(y + (((((int)blockIdx.x) * 1024) + (i_1 * 256)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    out_reg[i_2] = (x_reg[i_2] + y_reg[i_2]);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(float4*)(output + (((((int)blockIdx.x) * 1024) + (i_3 * 256)) + (((int)threadIdx.x) * 4))) = *(float4*)(out_reg + (i_3 * 4));
  }
}

