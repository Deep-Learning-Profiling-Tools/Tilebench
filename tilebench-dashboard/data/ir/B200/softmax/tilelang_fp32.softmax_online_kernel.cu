#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void softmax_online_kernel_kernel(const float* __restrict__ x, float* __restrict__ y, int M);
extern "C" __global__ void __launch_bounds__(128, 1) softmax_online_kernel_kernel(const float* __restrict__ x, float* __restrict__ y, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 512));
  float l[1];
  float running_max[1];
  float past_max[1];
  float X_local[16];
  float x_exp_local[16];
  float tile_max[1];
  float tile_sum_local[1];
  float Y_local[16];
  l[0] = 0x0p+0f/*0.000000e+00*/;
  running_max[0] = -CUDART_INF_F;
  past_max[0] = -CUDART_INF_F;
  __syncthreads();
  for (int _tmp = 0; _tmp < 5; ++_tmp) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = -CUDART_INF_F;
      *(float4*)(X_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(x_exp_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      *(ulonglong4*)(X_local + (i_2 * 8)) = tl::load_global_256(&(*(ulonglong4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp) * (int64_t)2048)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
    }
    tile_max[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      tile_max[0] = max(tile_max[0], X_local[(((rv & 1) * 8) + (rv >> 1))]);
    }
    tile_max[0] = tl::AllReduce<tl::MaxOp, 128, 1, 0, tl::NamedBarrier<128>>::run(tile_max[0], (&(((float*)workspace_1)[0])));
    running_max[0] = max(running_max[0], tile_max[0]);
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      x_exp_local[i_3] = expf((X_local[i_3] - running_max[0]));
    }
    tile_sum_local[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      tile_sum_local[0] = (tile_sum_local[0] + x_exp_local[(((rv_1 & 1) * 8) + (rv_1 >> 1))]);
    }
    tile_sum_local[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(tile_sum_local[0], (&(((float*)workspace)[0])));
    l[0] = ((l[0] * expf((past_max[0] - running_max[0]))) + tile_sum_local[0]);
    past_max[0] = running_max[0];
  }
  for (int _tmp_1 = 0; _tmp_1 < 5; ++_tmp_1) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(ulonglong4*)(X_local + (i_4 * 8)) = tl::load_global_256(&(*(ulonglong4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp_1) * (int64_t)2048)) + (((int64_t)i_4) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
    }
    float coef = (0x1p+0f/*1.000000e+00*/ / l[0]);
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      Y_local[i_5] = (expf((X_local[i_5] - running_max[0])) * coef);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::store_global_256(&(*(ulonglong4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp_1) * (int64_t)2048)) + (((int64_t)i_6) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))), *(ulonglong4*)(Y_local + (i_6 * 8)));
    }
  }
}

