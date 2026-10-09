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

extern "C" __global__ void layernorm_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, const half_t* __restrict__ bias, const half_t* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(128, 1) layernorm_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, const half_t* __restrict__ bias, const half_t* __restrict__ weight, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float sum_local[32];
  float sumsq_local[32];
  float sum_row[1];
  float sumsq_row[1];
  float mean_row[1];
  float rstd_row[1];
  half_t Y_local[32];
  half_t X_local_cast[4];
  half_t weight_local_cast_1[4];
  half_t bias_local_cast_2[4];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sum_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int tile = 0; tile < 3; ++tile) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      half_t broadcast_var_2 = half_t(0x0p+0f/*0.000000e+00*/);
      float4 __1;
      uint2 condval;
      if ((((tile * 2) + (i_2 >> 2)) < 5)) {
        condval = *(uint2*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      } else {
        condval = make_uint2(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
      }
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&condval))[0]);
      ((float2*)(&__1))[1] = __half22float2(((half2*)(&condval))[1]);
      float4 x_val = __1;
      float4 __2;
        float4 v_ = *(float4*)(sum_local + (i_2 * 4));
        __2.x = (v_.x+x_val.x);
        __2.y = (v_.y+x_val.y);
        __2.z = (v_.z+x_val.z);
        __2.w = (v_.w+x_val.w);
      *(float4*)(sum_local + (i_2 * 4)) = __2;
      float4 __3;
        float4 v__1 = *(float4*)(sumsq_local + (i_2 * 4));
        float4 __4;
          __4.x = (x_val.x*x_val.x);
          __4.y = (x_val.y*x_val.y);
          __4.z = (x_val.z*x_val.z);
          __4.w = (x_val.w*x_val.w);
        __3.x = (v__1.x+__4.x);
        __3.y = (v__1.y+__4.y);
        __3.z = (v__1.z+__4.z);
        __3.w = (v__1.w+__4.w);
      *(float4*)(sumsq_local + (i_2 * 4)) = __3;
    }
  }
  sum_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 32; ++rv) {
    sum_row[0] = (sum_row[0] + sum_local[(((rv & 7) * 4) + (rv >> 3))]);
  }
  sum_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sum_row[0], (&(((float*)workspace_1)[0])));
  sumsq_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
    sumsq_row[0] = (sumsq_row[0] + sumsq_local[(((rv_1 & 7) * 4) + (rv_1 >> 3))]);
  }
  __syncthreads();
  sumsq_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sumsq_row[0], (&(((float*)workspace)[0])));
  mean_row[0] = (sum_row[0] / 0x1.4p+13f/*1.024000e+04*/);
  float variance = ((sumsq_row[0] / 0x1.4p+13f/*1.024000e+04*/) - (mean_row[0] * mean_row[0]));
  rstd_row[0] = rsqrtf((variance + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  for (int tile_1 = 0; tile_1 < 3; ++tile_1) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_1;
      if ((((tile_1 * 2) + (i_3 >> 2)) < 5)) {
        condval_1 = *(uint2*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_3) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      } else {
        condval_1 = make_uint2(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
      }
      *(uint2*)(X_local_cast + 0) = condval_1;
      half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_2;
      if ((((tile_1 * 2) + (i_3 >> 2)) < 5)) {
        condval_2 = *(uint2*)(weight + (((tile_1 * 4096) + (i_3 * 512)) + (((int)threadIdx.x) * 4)));
      } else {
        condval_2 = make_uint2(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
      }
      *(uint2*)(weight_local_cast_1 + 0) = condval_2;
      half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_3;
      if ((((tile_1 * 2) + (i_3 >> 2)) < 5)) {
        condval_3 = *(uint2*)(bias + (((tile_1 * 4096) + (i_3 * 512)) + (((int)threadIdx.x) * 4)));
      } else {
        condval_3 = make_uint2(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
      *(uint2*)(bias_local_cast_2 + 0) = condval_3;
      uint2 __5;
      float4 __6;
        float4 __7;
          float4 __8;
            float4 __9;
              float4 __10;
              uint2 v__2 = *(uint2*)(X_local_cast + 0);
              ((float2*)(&__10))[0] = __half22float2(((half2*)(&v__2))[0]);
              ((float2*)(&__10))[1] = __half22float2(((half2*)(&v__2))[1]);
              float4 v__3 = make_float4(mean_row[0], mean_row[0], mean_row[0], mean_row[0]);
              __9.x = (__10.x-v__3.x);
              __9.y = (__10.y-v__3.y);
              __9.z = (__10.z-v__3.z);
              __9.w = (__10.w-v__3.w);
            float4 v__4 = make_float4(rstd_row[0], rstd_row[0], rstd_row[0], rstd_row[0]);
            __8.x = (__9.x*v__4.x);
            __8.y = (__9.y*v__4.y);
            __8.z = (__9.z*v__4.z);
            __8.w = (__9.w*v__4.w);
          float4 __11;
          uint2 v__5 = *(uint2*)(weight_local_cast_1 + 0);
          ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__5))[0]);
          ((float2*)(&__11))[1] = __half22float2(((half2*)(&v__5))[1]);
          __7.x = (__8.x*__11.x);
          __7.y = (__8.y*__11.y);
          __7.z = (__8.z*__11.z);
          __7.w = (__8.w*__11.w);
        float4 __12;
        uint2 v__6 = *(uint2*)(bias_local_cast_2 + 0);
        ((float2*)(&__12))[0] = __half22float2(((half2*)(&v__6))[0]);
        ((float2*)(&__12))[1] = __half22float2(((half2*)(&v__6))[1]);
        __6.x = (__7.x+__12.x);
        __6.y = (__7.y+__12.y);
        __6.z = (__7.z+__12.z);
        __6.w = (__7.w+__12.w);
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&__6))[1]);
      *(uint2*)(Y_local + (i_3 * 4)) = __5;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      if (((tile_1 * 2) + (i_4 >> 2)) < 5) {
        *(uint2*)(Y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_4) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(Y_local + (i_4 * 4));
      }
    }
  }
}

