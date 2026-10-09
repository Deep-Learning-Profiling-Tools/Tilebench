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
  bfloat16_t x_local_cast_1[16];
  bfloat16_t output_local_cast[16];
  bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  ulonglong4 condval;
  if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < 390625)) {
    condval = tl::load_global_256(&(*(ulonglong4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)))));
  } else {
    condval = make_ulonglong4(tl::pack_bfloat16x4(broadcast_var, broadcast_var, broadcast_var, broadcast_var), tl::pack_bfloat16x4(broadcast_var, broadcast_var, broadcast_var, broadcast_var), tl::pack_bfloat16x4(broadcast_var, broadcast_var, broadcast_var, broadcast_var), tl::pack_bfloat16x4(broadcast_var, broadcast_var, broadcast_var, broadcast_var));
  }
  *(ulonglong4*)(x_local_cast_1 + 0) = condval;
  for (int i = 0; i < 4; ++i) {
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
          uint2 v__3 = *(uint2*)(x_local_cast_1 + (i * 4));
          ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
          ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
          *(float2*)(&(__5.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(__6.x)));
          *(float2*)(&(__5.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(__6.z)));
        __4.x = expf(__5.x);
        __4.y = expf(__5.y);
        __4.z = expf(__5.z);
        __4.w = expf(__5.w);
        *(float2*)(&(__3.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(__4.x)));
        *(float2*)(&(__3.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(__4.z)));
      __2.x = (v_.x/__3.x);
      __2.y = (v_.y/__3.y);
      __2.z = (v_.z/__3.z);
      __2.w = (v_.w/__3.w);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
    *(uint2*)(output_local_cast + (i * 4)) = __1;
  }
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < 390625) {
    tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)))), *(ulonglong4*)(output_local_cast + 0));
  }
}

