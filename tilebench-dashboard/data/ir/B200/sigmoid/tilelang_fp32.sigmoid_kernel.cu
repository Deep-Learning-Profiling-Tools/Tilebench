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
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 390625) {
    float broadcast_var = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    ulonglong4 __1;
      ulonglong4 v_ = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var));
      ulonglong4 __2;
        ulonglong4 v__1 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1));
        ulonglong4 __3;
        ulonglong4 __4;
          ulonglong4 v__2 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2));
          ulonglong4 v__3 = tl::load_global_256(&(*(ulonglong4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
          *(float2*)(&(__4.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__4.y)) = tl::sub2(*(float2*)(&(v__2.y)), *(float2*)(&(v__3.y)));
          *(float2*)(&(__4.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
          *(float2*)(&(__4.w)) = tl::sub2(*(float2*)(&(v__2.w)), *(float2*)(&(v__3.w)));
        ((float2*)(&(__3.x)))->x = expf(((float2*)(&(__4.x)))->x);
        ((float2*)(&(__3.x)))->y = expf(((float2*)(&(__4.x)))->y);
        ((float2*)(&(__3.y)))->x = expf(((float2*)(&(__4.y)))->x);
        ((float2*)(&(__3.y)))->y = expf(((float2*)(&(__4.y)))->y);
        ((float2*)(&(__3.z)))->x = expf(((float2*)(&(__4.z)))->x);
        ((float2*)(&(__3.z)))->y = expf(((float2*)(&(__4.z)))->y);
        ((float2*)(&(__3.w)))->x = expf(((float2*)(&(__4.w)))->x);
        ((float2*)(&(__3.w)))->y = expf(((float2*)(&(__4.w)))->y);
        *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(__3.x)));
        *(float2*)(&(__2.y)) = tl::add2(*(float2*)(&(v__1.y)), *(float2*)(&(__3.y)));
        *(float2*)(&(__2.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(__3.z)));
        *(float2*)(&(__2.w)) = tl::add2(*(float2*)(&(v__1.w)), *(float2*)(&(__3.w)));
      ((float2*)(&(__1.x)))->x = (((float2*)(&(v_.x)))->x/((float2*)(&(__2.x)))->x);
      ((float2*)(&(__1.x)))->y = (((float2*)(&(v_.x)))->y/((float2*)(&(__2.x)))->y);
      ((float2*)(&(__1.y)))->x = (((float2*)(&(v_.y)))->x/((float2*)(&(__2.y)))->x);
      ((float2*)(&(__1.y)))->y = (((float2*)(&(v_.y)))->y/((float2*)(&(__2.y)))->y);
      ((float2*)(&(__1.z)))->x = (((float2*)(&(v_.z)))->x/((float2*)(&(__2.z)))->x);
      ((float2*)(&(__1.z)))->y = (((float2*)(&(v_.z)))->y/((float2*)(&(__2.z)))->y);
      ((float2*)(&(__1.w)))->x = (((float2*)(&(v_.w)))->x/((float2*)(&(__2.w)))->x);
      ((float2*)(&(__1.w)))->y = (((float2*)(&(v_.w)))->y/((float2*)(&(__2.w)))->y);
    tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))), __1);
  }
}

