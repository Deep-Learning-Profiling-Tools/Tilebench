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

extern "C" __global__ void weight_dequant_kernel_kernel(const half_t* __restrict__ S, const half_t* __restrict__ X, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) weight_dequant_kernel_kernel(const half_t* __restrict__ S, const half_t* __restrict__ X, half_t* __restrict__ output) {
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float value = (((float)X[((((((int)blockIdx.x) * 4096) + ((i >> 2) * 512)) + (((int)threadIdx.x) * 4)) + (i & 3))]) * ((float)S[((((((int)blockIdx.x) / 320) * 80) + ((((((int)blockIdx.x) * 8) + (i >> 2)) % 20) * 4)) + (((int)threadIdx.x) >> 5))]));
    output[((((((int)blockIdx.x) * 4096) + ((i >> 2) * 512)) + (((int)threadIdx.x) * 4)) + (i & 3))] = ((half_t)value);
  }
}

