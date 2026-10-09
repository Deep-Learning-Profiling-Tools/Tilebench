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

extern "C" __global__ void rmsnorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ rms_w, int M);
extern "C" __global__ void __launch_bounds__(256, 1) rmsnorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ rms_w, int M) {
  float sumsq_local[8];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  float inv_rms[1];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int tile = 0; tile < 5; ++tile) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float4 x_val = *(float4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)2048)) + (((int64_t)i_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      float4 __1;
        float4 v_ = *(float4*)(sumsq_local + (i_1 * 4));
        float4 __2;
          __2.x = (x_val.x*x_val.x);
          __2.y = (x_val.y*x_val.y);
          __2.z = (x_val.z*x_val.z);
          __2.w = (x_val.w*x_val.w);
        __1.x = (v_.x+__2.x);
        __1.y = (v_.y+__2.y);
        __1.z = (v_.z+__2.z);
        __1.w = (v_.w+__2.w);
      *(float4*)(sumsq_local + (i_1 * 4)) = __1;
    }
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    row_sum[0] = (row_sum[0] + sumsq_local[(((rv & 1) * 4) + (rv >> 1))]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(row_sum[0], (&(workspace[0])));
  inv_rms[0] = rsqrtf(((row_sum[0] / 0x1.4p+13f/*1.024000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 5; ++tile_1) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      float4 x_val_1 = *(float4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)2048)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      float4 weight = *(float4*)(rms_w + (((tile_1 * 2048) + (i_2 * 1024)) + (((int)threadIdx.x) * 4)));
      float4 __3;
        float4 __4;
          float4 v__1 = make_float4(inv_rms[0], inv_rms[0], inv_rms[0], inv_rms[0]);
          __4.x = (x_val_1.x*v__1.x);
          __4.y = (x_val_1.y*v__1.y);
          __4.z = (x_val_1.z*v__1.z);
          __4.w = (x_val_1.w*v__1.w);
        __3.x = (__4.x*weight.x);
        __3.y = (__4.y*weight.y);
        __3.z = (__4.z*weight.z);
        __3.w = (__4.w*weight.w);
      *(float4*)(Y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)2048)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = __3;
    }
  }
}

