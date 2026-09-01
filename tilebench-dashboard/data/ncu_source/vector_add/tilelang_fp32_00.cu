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
extern "C" __global__ void __launch_bounds__(256, 1) add_kernel_kernel(float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y) {
  float x_reg[4];
  float y_reg[4];
  float out_reg[4];
  *(float4*)(x_reg + 0) = *(float4*)(x + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  *(float4*)(y_reg + 0) = *(float4*)(y + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    out_reg[i] = (x_reg[i] + y_reg[i]);
  }
  *(float4*)(output + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(out_reg + 0);
}

