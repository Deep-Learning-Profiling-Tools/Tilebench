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

extern "C" __global__ void swiglu_kernel_kernel(float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y);
extern "C" __global__ void __launch_bounds__(256, 1) swiglu_kernel_kernel(float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y) {
  float4 x_f32 = *(float4*)(x + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  float4 y_f32 = *(float4*)(y + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  float broadcast_var = 0x1p+0f/*1.000000e+00*/;
  float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  float4 __1;
    float4 __2;
      float4 __3;
        float4 v_ = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        float4 __4;
          float4 v__1 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          float4 __5;
          float4 __6;
            float4 v__2 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            __6.x = (v__2.x-x_f32.x);
            __6.y = (v__2.y-x_f32.y);
            __6.z = (v__2.z-x_f32.z);
            __6.w = (v__2.w-x_f32.w);
          __5.x = expf(__6.x);
          __5.y = expf(__6.y);
          __5.z = expf(__6.z);
          __5.w = expf(__6.w);
          __4.x = (v__1.x+__5.x);
          __4.y = (v__1.y+__5.y);
          __4.z = (v__1.z+__5.z);
          __4.w = (v__1.w+__5.w);
        __3.x = (v_.x/__4.x);
        __3.y = (v_.y/__4.y);
        __3.z = (v_.z/__4.z);
        __3.w = (v_.w/__4.w);
      __2.x = (x_f32.x*__3.x);
      __2.y = (x_f32.y*__3.y);
      __2.z = (x_f32.z*__3.z);
      __2.w = (x_f32.w*__3.w);
    __1.x = (__2.x*y_f32.x);
    __1.y = (__2.y*y_f32.y);
    __1.z = (__2.z*y_f32.z);
    __1.w = (__2.w*y_f32.w);
  *(float4*)(output + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = __1;
}

