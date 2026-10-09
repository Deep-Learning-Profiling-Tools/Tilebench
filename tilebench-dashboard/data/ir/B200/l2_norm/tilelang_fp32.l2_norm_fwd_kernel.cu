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

extern "C" __global__ void l2_norm_fwd_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, int M);
extern "C" __global__ void __launch_bounds__(128, 1) l2_norm_fwd_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, int M) {
  float acc[4];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  float inv_norm[1];
  float Y_local[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(acc + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int tile = 0; tile < 20; ++tile) {
    float4 x_val = *(float4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    float4 __1;
      float4 v_ = *(float4*)(acc + 0);
      *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(x_val.x)), *(float2*)(&(x_val.x)), *(float2*)(&(v_.x)));
      *(float2*)(&(__1.z)) = tl::fma2(*(float2*)(&(x_val.z)), *(float2*)(&(x_val.z)), *(float2*)(&(v_.z)));
    *(float4*)(acc + 0) = __1;
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    row_sum[0] = (row_sum[0] + acc[rv]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(row_sum[0], (&(workspace[0])));
  inv_norm[0] = rsqrtf((row_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 20; ++tile_1) {
    float4 x_val_1 = *(float4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    float4 __2;
      float4 v__1 = make_float4(inv_norm[0], inv_norm[0], inv_norm[0], inv_norm[0]);
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(x_val_1.x)), *(float2*)(&(v__1.x)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(x_val_1.z)), *(float2*)(&(v__1.z)));
    *(float4*)(Y_local + 0) = __2;
    *(float4*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(float4*)(Y_local + 0);
  }
}

