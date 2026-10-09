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

extern "C" __global__ void weight_dequant_kernel_kernel(const float* __restrict__ S, const float* __restrict__ X, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) weight_dequant_kernel_kernel(const float* __restrict__ S, const float* __restrict__ X, float* __restrict__ output) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float4 __1;
      float4 v_ = *(float4*)(X + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
      float4 v__1 = make_float4(S[(((((((int)blockIdx.x) / 1280) * 80) + ((((int)blockIdx.x) % 10) * 8)) + (i * 4)) + (((int)threadIdx.x) >> 5))], S[(((((((int)blockIdx.x) / 1280) * 80) + ((((int)blockIdx.x) % 10) * 8)) + (i * 4)) + (((int)threadIdx.x) >> 5))], S[(((((((int)blockIdx.x) / 1280) * 80) + ((((int)blockIdx.x) % 10) * 8)) + (i * 4)) + (((int)threadIdx.x) >> 5))], S[(((((((int)blockIdx.x) / 1280) * 80) + ((((int)blockIdx.x) % 10) * 8)) + (i * 4)) + (((int)threadIdx.x) >> 5))]);
      __1.x = (v_.x*v__1.x);
      __1.y = (v_.y*v__1.y);
      __1.z = (v_.z*v__1.z);
      __1.w = (v_.w*v__1.w);
    float4 value = __1;
    *(float4*)(output + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4))) = value;
  }
}

