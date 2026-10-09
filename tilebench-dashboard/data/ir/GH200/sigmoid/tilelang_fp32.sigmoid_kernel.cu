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

extern "C" __global__ void sigmoid_kernel_kernel(float* __restrict__ output, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) sigmoid_kernel_kernel(float* __restrict__ output, const float* __restrict__ x) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 5)) < 390625) {
    float broadcast_var = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    float4 __1;
      float4 v_ = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      float4 __2;
        float4 v__1 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        float4 __3;
        float4 __4;
          float4 v__2 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          float4 v__3 = *(float4*)(x + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
          __4.x = (v__2.x-v__3.x);
          __4.y = (v__2.y-v__3.y);
          __4.z = (v__2.z-v__3.z);
          __4.w = (v__2.w-v__3.w);
        __3.x = expf(__4.x);
        __3.y = expf(__4.y);
        __3.z = expf(__4.z);
        __3.w = expf(__4.w);
        __2.x = (v__1.x+__3.x);
        __2.y = (v__1.y+__3.y);
        __2.z = (v__1.z+__3.z);
        __2.w = (v__1.w+__3.w);
      __1.x = (v_.x/__2.x);
      __1.y = (v_.y/__2.y);
      __1.z = (v_.z/__2.z);
      __1.w = (v_.w/__2.w);
    *(float4*)(output + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = __1;
  }
}

