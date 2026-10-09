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

extern "C" __global__ void mean_reduction_kernel_kernel(float* __restrict__ output, const bfloat16_t* __restrict__ x, int M);
extern "C" __global__ void __launch_bounds__(256, 1) mean_reduction_kernel_kernel(float* __restrict__ output, const bfloat16_t* __restrict__ x, int M) {
  float acc[8];
  bfloat16_t x_local_cast[8];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int tile_idx = 0; tile_idx < 10; ++tile_idx) {
    *(uint4*)(x_local_cast + 0) = *(uint4*)(x + (((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)tile_idx) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float4 __1;
        float4 v_ = *(float4*)(acc + (i_1 * 4));
        float4 __2;
        uint2 v__1 = *(uint2*)(x_local_cast + (i_1 * 4));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        __1.x = (v_.x+__2.x);
        __1.y = (v_.y+__2.y);
        __1.z = (v_.z+__2.z);
        __1.w = (v_.w+__2.w);
      *(float4*)(acc + (i_1 * 4)) = __1;
    }
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    row_sum[0] = (row_sum[0] + acc[rv]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(row_sum[0], (&(workspace[0])));
  output[((int64_t)((int)blockIdx.x))] = (row_sum[0] / 0x1.4p+14f/*2.048000e+04*/);
}

