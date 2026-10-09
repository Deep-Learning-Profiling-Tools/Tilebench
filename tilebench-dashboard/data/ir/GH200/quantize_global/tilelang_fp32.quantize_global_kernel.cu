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

extern "C" __global__ void quantize_global_kernel_kernel(half_t* __restrict__ output, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) quantize_global_kernel_kernel(half_t* __restrict__ output, const float* __restrict__ x) {
  float x_local_cast_1[4];
  half_t output_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(x_local_cast_1 + 0) = *(float4*)(x + (((((int)blockIdx.x) * 8192) + (i * 512)) + (((int)threadIdx.x) * 4)));
    uint2 __1;
    float4 v_ = *(float4*)(x_local_cast_1 + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(output_local_cast + 0) = __1;
    *(uint2*)(output + (((((int)blockIdx.x) * 8192) + (i * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(output_local_cast + 0);
  }
}

