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

extern "C" __global__ void dropout_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ x_keep);
extern "C" __global__ void __launch_bounds__(64, 1) dropout_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ x_keep) {
  half_t x_reg[16];
  half_t x_keep_reg[16];
  half_t output_reg[16];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(x_reg + (i * 8)) = *(uint4*)(x + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(x_keep_reg + (i_1 * 8)) = *(uint4*)(x_keep + (((((int)blockIdx.x) * 1024) + (i_1 * 512)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    output_reg[i_2] = ((x_keep_reg[i_2] != half_t(0x0p+0f/*0.000000e+00*/)) ? (x_reg[i_2] * half_t(0x1p+1f/*2.000000e+00*/)) : half_t(0x0p+0f/*0.000000e+00*/));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(uint4*)(output + (((((int)blockIdx.x) * 1024) + (i_3 * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_reg + (i_3 * 8));
  }
}

