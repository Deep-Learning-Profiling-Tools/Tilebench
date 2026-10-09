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

extern "C" __global__ void mul2_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) mul2_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x) {
  bfloat16_t x_reg[8];
  bfloat16_t output_reg[8];
  *(uint4*)(x_reg + 0) = *(uint4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    output_reg[i] = (x_reg[i] * bfloat16_t(0x1p+1f/*2.000000e+00*/));
  }
  *(uint4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_reg + 0);
}

