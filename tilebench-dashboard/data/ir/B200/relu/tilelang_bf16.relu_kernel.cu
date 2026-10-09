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

extern "C" __global__ void relu_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) relu_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x) {
  bfloat16_t x_reg[16];
  bfloat16_t output_reg[16];
  *(ulonglong4*)(x_reg + 0) = tl::load_global_256(&(*(ulonglong4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)))));
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    bfloat16_t value = x_reg[i];
    output_reg[i] = ((bfloat16_t(0x0p+0f/*0.000000e+00*/) <= value) ? value : bfloat16_t(0x0p+0f/*0.000000e+00*/));
  }
  tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)))), *(ulonglong4*)(output_reg + 0));
}

