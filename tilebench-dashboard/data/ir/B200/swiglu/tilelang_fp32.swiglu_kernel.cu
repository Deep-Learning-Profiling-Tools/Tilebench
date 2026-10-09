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
  ulonglong4 x_f32 = tl::load_global_256(&(*(ulonglong4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
  ulonglong4 y_f32 = tl::load_global_256(&(*(ulonglong4*)(y + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
  float broadcast_var = 0x1p+0f/*1.000000e+00*/;
  float broadcast_var_1 = 0x1p+0f/*1.000000e+00*/;
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  ulonglong4 __1;
    ulonglong4 __2;
      ulonglong4 __3;
        ulonglong4 v_ = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var));
        ulonglong4 __4;
          ulonglong4 v__1 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1));
          ulonglong4 __5;
          ulonglong4 __6;
            ulonglong4 v__2 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2));
            *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(x_f32.x)));
            *(float2*)(&(__6.y)) = tl::sub2(*(float2*)(&(v__2.y)), *(float2*)(&(x_f32.y)));
            *(float2*)(&(__6.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(x_f32.z)));
            *(float2*)(&(__6.w)) = tl::sub2(*(float2*)(&(v__2.w)), *(float2*)(&(x_f32.w)));
          ((float2*)(&(__5.x)))->x = expf(((float2*)(&(__6.x)))->x);
          ((float2*)(&(__5.x)))->y = expf(((float2*)(&(__6.x)))->y);
          ((float2*)(&(__5.y)))->x = expf(((float2*)(&(__6.y)))->x);
          ((float2*)(&(__5.y)))->y = expf(((float2*)(&(__6.y)))->y);
          ((float2*)(&(__5.z)))->x = expf(((float2*)(&(__6.z)))->x);
          ((float2*)(&(__5.z)))->y = expf(((float2*)(&(__6.z)))->y);
          ((float2*)(&(__5.w)))->x = expf(((float2*)(&(__6.w)))->x);
          ((float2*)(&(__5.w)))->y = expf(((float2*)(&(__6.w)))->y);
          *(float2*)(&(__4.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(__5.x)));
          *(float2*)(&(__4.y)) = tl::add2(*(float2*)(&(v__1.y)), *(float2*)(&(__5.y)));
          *(float2*)(&(__4.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(__5.z)));
          *(float2*)(&(__4.w)) = tl::add2(*(float2*)(&(v__1.w)), *(float2*)(&(__5.w)));
        ((float2*)(&(__3.x)))->x = (((float2*)(&(v_.x)))->x/((float2*)(&(__4.x)))->x);
        ((float2*)(&(__3.x)))->y = (((float2*)(&(v_.x)))->y/((float2*)(&(__4.x)))->y);
        ((float2*)(&(__3.y)))->x = (((float2*)(&(v_.y)))->x/((float2*)(&(__4.y)))->x);
        ((float2*)(&(__3.y)))->y = (((float2*)(&(v_.y)))->y/((float2*)(&(__4.y)))->y);
        ((float2*)(&(__3.z)))->x = (((float2*)(&(v_.z)))->x/((float2*)(&(__4.z)))->x);
        ((float2*)(&(__3.z)))->y = (((float2*)(&(v_.z)))->y/((float2*)(&(__4.z)))->y);
        ((float2*)(&(__3.w)))->x = (((float2*)(&(v_.w)))->x/((float2*)(&(__4.w)))->x);
        ((float2*)(&(__3.w)))->y = (((float2*)(&(v_.w)))->y/((float2*)(&(__4.w)))->y);
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(x_f32.x)), *(float2*)(&(__3.x)));
      *(float2*)(&(__2.y)) = tl::mul2(*(float2*)(&(x_f32.y)), *(float2*)(&(__3.y)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(x_f32.z)), *(float2*)(&(__3.z)));
      *(float2*)(&(__2.w)) = tl::mul2(*(float2*)(&(x_f32.w)), *(float2*)(&(__3.w)));
    *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(__2.x)), *(float2*)(&(y_f32.x)));
    *(float2*)(&(__1.y)) = tl::mul2(*(float2*)(&(__2.y)), *(float2*)(&(y_f32.y)));
    *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(__2.z)), *(float2*)(&(y_f32.z)));
    *(float2*)(&(__1.w)) = tl::mul2(*(float2*)(&(__2.w)), *(float2*)(&(y_f32.w)));
  tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))), __1);
}

