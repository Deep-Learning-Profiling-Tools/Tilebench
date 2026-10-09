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

extern "C" __global__ void main_kernel(float* __restrict__ work, int j, int k);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ work, int j, int k) {
  float a = 0x0p+0f/*0.000000e+00*/;
  float b = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    bool active = (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) < ((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j)) && (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j) < 16777216));
    a = 0x0p+0f/*0.000000e+00*/;
    b = 0x0p+0f/*0.000000e+00*/;
    if (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) < ((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j)) && (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j) < 16777216)) {
      a = work[(((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x))];
      b = work[((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)128)) + ((int64_t)((int)threadIdx.x))) ^ ((int64_t)j))];
    }
    bool ascending = (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) & k) == 0);
    bool swap = ((((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) & k) == 0) ? (b < a) : (a < b));
    float new_a = (swap ? b : a);
    float new_b = (swap ? a : b);
    if (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) < ((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j)) && (((((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x)) ^ j) < 16777216)) {
      work[(((((int)blockIdx.x) * 512) + (i * 128)) + ((int)threadIdx.x))] = new_a;
      work[((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)128)) + ((int64_t)((int)threadIdx.x))) ^ ((int64_t)j))] = new_b;
    }
  }
}

