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

extern "C" __global__ void softmax_online_kernel_kernel(const half_t* __restrict__ x, half_t* __restrict__ y, int M);
extern "C" __global__ void __launch_bounds__(128, 1) softmax_online_kernel_kernel(const half_t* __restrict__ x, half_t* __restrict__ y, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 512));
  float l[1];
  float running_max[1];
  float past_max[1];
  float X_local[16];
  float x_exp_local[16];
  half_t x_local_cast[16];
  float tile_max[1];
  float tile_sum_local[1];
  half_t x_local_cast_1[16];
  half_t Y_local[16];
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
    *(ulonglong4*)(x_local_cast + 0) = tl::load_global_256(&(*(ulonglong4*)(x + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)))));
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float4 __1;
      uint2 v_ = *(uint2*)(x_local_cast + (i_2 * 4));
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      ((float2*)(&__1))[1] = __half22float2(((half2*)(&v_))[1]);
      *(float4*)(X_local + (i_2 * 4)) = __1;
    }
    tile_max[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      tile_max[0] = max(tile_max[0], X_local[rv]);
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
      tile_sum_local[0] = (tile_sum_local[0] + x_exp_local[rv_1]);
    }
    tile_sum_local[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(tile_sum_local[0], (&(((float*)workspace)[0])));
    l[0] = ((l[0] * expf((past_max[0] - running_max[0]))) + tile_sum_local[0]);
    past_max[0] = running_max[0];
  }
  for (int _tmp_1 = 0; _tmp_1 < 5; ++_tmp_1) {
    *(ulonglong4*)(x_local_cast_1 + 0) = tl::load_global_256(&(*(ulonglong4*)(x + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)))));
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float4 __2;
      uint2 v__1 = *(uint2*)(x_local_cast_1 + (i_4 * 4));
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      ((float2*)(&__2))[1] = __half22float2(((half2*)(&v__1))[1]);
      *(float4*)(X_local + (i_4 * 4)) = __2;
    }
    float coef = (0x1p+0f/*1.000000e+00*/ / l[0]);
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      uint2 __3;
      float4 __4;
        float4 __5;
        float4 __6;
          float4 v__2 = *(float4*)(X_local + (i_5 * 4));
          float4 v__3 = make_float4(running_max[0], running_max[0], running_max[0], running_max[0]);
          *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__6.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
        __5.x = expf(__6.x);
        __5.y = expf(__6.y);
        __5.z = expf(__6.z);
        __5.w = expf(__6.w);
        float4 v__4 = make_float4(coef, coef, coef, coef);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(__5.x)), *(float2*)(&(v__4.x)));
        *(float2*)(&(__4.z)) = tl::mul2(*(float2*)(&(__5.z)), *(float2*)(&(v__4.z)));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&__4))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&__4))[1]);
      *(uint2*)(Y_local + (i_5 * 4)) = __3;
    }
    tl::store_global_256(&(*(ulonglong4*)(y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)_tmp_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)))), *(ulonglong4*)(Y_local + 0));
  }
}

