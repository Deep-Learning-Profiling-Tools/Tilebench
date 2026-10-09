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

extern "C" __global__ void swiglu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ y);
extern "C" __global__ void __launch_bounds__(128, 1) swiglu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x, const half_t* __restrict__ y) {
  half_t x_local_cast_1[4];
  half_t y_local_cast_2[4];
  half_t output_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint2*)(x_local_cast_1 + 0) = *(uint2*)(x + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    *(uint2*)(y_local_cast_2 + 0) = *(uint2*)(y + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    float broadcast_var = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    uint2 __1;
    float4 __2;
      float4 __3;
        float4 __4;
        uint2 v_ = *(uint2*)(x_local_cast_1 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__4))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 __5;
          float4 v__1 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          float4 __6;
            float4 v__2 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
            float4 __7;
            float4 __8;
              float4 v__3 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
              float4 __9;
              ((float2*)(&__9))[0] = __half22float2(((half2*)(&v_))[0]);
              ((float2*)(&__9))[1] = __half22float2(((half2*)(&v_))[1]);
              __8.x = (v__3.x-__9.x);
              __8.y = (v__3.y-__9.y);
              __8.z = (v__3.z-__9.z);
              __8.w = (v__3.w-__9.w);
            __7.x = expf(__8.x);
            __7.y = expf(__8.y);
            __7.z = expf(__8.z);
            __7.w = expf(__8.w);
            __6.x = (v__2.x+__7.x);
            __6.y = (v__2.y+__7.y);
            __6.z = (v__2.z+__7.z);
            __6.w = (v__2.w+__7.w);
          __5.x = (v__1.x/__6.x);
          __5.y = (v__1.y/__6.y);
          __5.z = (v__1.z/__6.z);
          __5.w = (v__1.w/__6.w);
        __3.x = (__4.x*__5.x);
        __3.y = (__4.y*__5.y);
        __3.z = (__4.z*__5.z);
        __3.w = (__4.w*__5.w);
      float4 __10;
      uint2 v__4 = *(uint2*)(y_local_cast_2 + 0);
      ((float2*)(&__10))[0] = __half22float2(((half2*)(&v__4))[0]);
      ((float2*)(&__10))[1] = __half22float2(((half2*)(&v__4))[1]);
      __2.x = (__3.x*__10.x);
      __2.y = (__3.y*__10.y);
      __2.z = (__3.z*__10.z);
      __2.w = (__3.w*__10.w);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
    *(uint2*)(output_local_cast + 0) = __1;
    *(uint2*)(output + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(output_local_cast + 0);
  }
}

