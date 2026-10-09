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

extern "C" __global__ void dequantize_rowwise_kernel_kernel(half_t* __restrict__ output, const float* __restrict__ state_x, const signed char* __restrict__ x, int rows);
extern "C" __global__ void __launch_bounds__(128, 1) dequantize_rowwise_kernel_kernel(half_t* __restrict__ output, const float* __restrict__ state_x, const signed char* __restrict__ x, int rows) {
  signed char x_local[8];
  half_t y_local[8];
  float scale = (state_x[((int64_t)((int)blockIdx.x))] * 0x1.0204081020408p-7f/*7.874016e-03*/);
  *(int2*)(x_local + 0) = *(int2*)(x + (((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + (((int64_t)((int)blockIdx.y)) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint2 __1;
    float4 __2;
      float4 __3;
      int v_ = *(int*)(x_local + (i * 4));
      __3.x = (float)(((char)(v_ >> 0)));
      __3.y = (float)(((char)(v_ >> 8)));
      __3.z = (float)(((char)(v_ >> 16)));
      __3.w = (float)(((char)(v_ >> 24)));
      float4 v__1 = make_float4(scale, scale, scale, scale);
      __2.x = (__3.x*v__1.x);
      __2.y = (__3.y*v__1.y);
      __2.z = (__3.z*v__1.z);
      __2.w = (__3.w*v__1.w);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
    *(uint2*)(y_local + (i * 4)) = __1;
  }
  *(uint4*)(output + (((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + (((int64_t)((int)blockIdx.y)) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(y_local + 0);
}

