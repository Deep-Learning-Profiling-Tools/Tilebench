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

extern "C" __global__ void relu_kernel_kernel(signed char* __restrict__ output, const signed char* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) relu_kernel_kernel(signed char* __restrict__ output, const signed char* __restrict__ x) {
  signed char x_reg[16];
  signed char output_reg[16];
  *(int4*)(x_reg + 0) = *(int4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)));
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    signed char value = x_reg[i];
    output_reg[i] = (((signed char)0 <= value) ? value : (signed char)0);
  }
  *(int4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16))) = *(int4*)(output_reg + 0);
}

