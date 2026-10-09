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

extern "C" __global__ void sigmoid_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) sigmoid_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x) {
  bfloat16_t x_local_cast_1[8];
  bfloat16_t output_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((int)blockIdx.x) * 16) + (i * 8)) + (((int)threadIdx.x) >> 4)) < 390625)) {
      condval = *(uint4*)(x + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(x_local_cast_1 + 0) = condval;
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
      float broadcast_var_2 = 0x1p+0f/*1.000000e+00*/;
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      uint2 __1;
      float4 __2;
        float4 v_ = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        float4 __3;
          float4 v__1 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          float4 __4;
          float4 __5;
            float4 v__2 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            float4 __6;
            uint2 v__3 = *(uint2*)(x_local_cast_1 + (vec * 4));
            ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
            __5.x = (v__2.x-__6.x);
            __5.y = (v__2.y-__6.y);
            __5.z = (v__2.z-__6.z);
            __5.w = (v__2.w-__6.w);
          __4.x = expf(__5.x);
          __4.y = expf(__5.y);
          __4.z = expf(__5.z);
          __4.w = expf(__5.w);
          __3.x = (v__1.x+__4.x);
          __3.y = (v__1.y+__4.y);
          __3.z = (v__1.z+__4.z);
          __3.w = (v__1.w+__4.w);
        __2.x = (v_.x/__3.x);
        __2.y = (v_.y/__3.y);
        __2.z = (v_.z/__3.z);
        __2.w = (v_.w/__3.w);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
      *(uint2*)(output_local_cast + (vec * 4)) = __1;
    }
    if ((((((int)blockIdx.x) * 16) + (i * 8)) + (((int)threadIdx.x) >> 4)) < 390625) {
      *(uint4*)(output + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_local_cast + 0);
    }
  }
}

