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

extern "C" __global__ void relu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(64, 1) relu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x) {
  half_t x_reg[32];
  half_t output_reg[32];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(ulonglong4*)(x_reg + (i * 16)) = tl::load_global_256(&(*(ulonglong4*)(x + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 16)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    half_t value = x_reg[i_1];
    output_reg[i_1] = ((half_t(0x0p+0f/*0.000000e+00*/) <= value) ? value : half_t(0x0p+0f/*0.000000e+00*/));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::store_global_256(&(*(ulonglong4*)(output + (((((int)blockIdx.x) * 2048) + (i_2 * 1024)) + (((int)threadIdx.x) * 16)))), *(ulonglong4*)(output_reg + (i_2 * 16)));
  }
}

