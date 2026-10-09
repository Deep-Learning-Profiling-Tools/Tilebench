#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void pad_kernel_kernel(const float* __restrict__ data, float* __restrict__ work);
extern "C" __global__ void __launch_bounds__(128, 1) pad_kernel_kernel(const float* __restrict__ data, float* __restrict__ work) {
  float broadcast_var = CUDART_INF_F;
  float4 condval;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 78125)) {
    condval = *(float4*)(data + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)));
  } else {
    condval = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(float4*)(work + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))) = condval;
}

