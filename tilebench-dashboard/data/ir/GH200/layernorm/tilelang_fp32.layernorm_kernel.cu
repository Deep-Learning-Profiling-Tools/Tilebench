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

extern "C" __global__ void layernorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ bias, const float* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(256, 1) layernorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ bias, const float* __restrict__ weight, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float sum_local[2];
  float sumsq_local[2];
  float sum_row[1];
  float sumsq_row[1];
  float mean_row[1];
  float rstd_row[1];
  float Y_local[2];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(sum_local + 0) = make_float2(broadcast_var, broadcast_var);
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(sumsq_local + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  for (int tile = 0; tile < 20; ++tile) {
    float2 x_val = *(float2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
    float2 __1;
      float2 v_ = *(float2*)(sum_local + 0);
      __1.x = (v_.x+x_val.x);
      __1.y = (v_.y+x_val.y);
    *(float2*)(sum_local + 0) = __1;
    float2 __2;
      float2 v__1 = *(float2*)(sumsq_local + 0);
      float2 __3;
        __3.x = (x_val.x*x_val.x);
        __3.y = (x_val.y*x_val.y);
      __2.x = (v__1.x+__3.x);
      __2.y = (v__1.y+__3.y);
    *(float2*)(sumsq_local + 0) = __2;
  }
  sum_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    sum_row[0] = (sum_row[0] + sum_local[rv]);
  }
  sum_row[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(sum_row[0], (&(((float*)workspace_1)[0])));
  sumsq_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
    sumsq_row[0] = (sumsq_row[0] + sumsq_local[rv_1]);
  }
  __syncthreads();
  sumsq_row[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(sumsq_row[0], (&(((float*)workspace)[0])));
  mean_row[0] = (sum_row[0] / 0x1.4p+13f/*1.024000e+04*/);
  float variance = ((sumsq_row[0] / 0x1.4p+13f/*1.024000e+04*/) - (mean_row[0] * mean_row[0]));
  rstd_row[0] = rsqrtf((variance + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  for (int tile_1 = 0; tile_1 < 20; ++tile_1) {
    float2 x_val_1 = *(float2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
    float2 __4;
      float2 __5;
        float2 v__2 = make_float2(mean_row[0], mean_row[0]);
        __5.x = (x_val_1.x-v__2.x);
        __5.y = (x_val_1.y-v__2.y);
      float2 v__3 = make_float2(rstd_row[0], rstd_row[0]);
      __4.x = (__5.x*v__3.x);
      __4.y = (__5.y*v__3.y);
    float2 norm_1 = __4;
    float2 __6;
      float2 __7;
        float2 v__4 = *(float2*)(weight + ((tile_1 * 512) + (((int)threadIdx.x) * 2)));
        __7.x = (norm_1.x*v__4.x);
        __7.y = (norm_1.y*v__4.y);
      float2 v__5 = *(float2*)(bias + ((tile_1 * 512) + (((int)threadIdx.x) * 2)));
      __6.x = (__7.x+v__5.x);
      __6.y = (__7.y+v__5.y);
    *(float2*)(Y_local + 0) = __6;
    *(float2*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2))) = *(float2*)(Y_local + 0);
  }
}

