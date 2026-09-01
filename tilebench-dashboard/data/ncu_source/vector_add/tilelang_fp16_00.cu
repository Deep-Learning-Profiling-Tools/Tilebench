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

extern "C" __global__ void add_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ y);
extern "C" __global__ void __launch_bounds__(64, 1) add_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ y) {
  half_t x_reg[32];
  half_t y_reg[32];
  half_t out_reg[32];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(x_reg + (i * 8)) = *(uint4*)(x + (((((int)blockIdx.x) * 2048) + (i * 512)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(y_reg + (i_1 * 8)) = *(uint4*)(y + (((((int)blockIdx.x) * 2048) + (i_1 * 512)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    out_reg[i_2] = (x_reg[i_2] + y_reg[i_2]);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(uint4*)(output + (((((int)blockIdx.x) * 2048) + (i_3 * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(out_reg + (i_3 * 8));
  }
}

