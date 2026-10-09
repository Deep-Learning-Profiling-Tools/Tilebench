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
  float sum_local[8];
  float sumsq_local[8];
  float sum_row[1];
  float sumsq_row[1];
  float mean_row[1];
  float rstd_row[1];
  half_t X_local_cast[8];
  half_t weight_local_cast_1[8];
  half_t bias_local_cast_2[8];
  half_t Y_local[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sum_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int tile = 0; tile < 10; ++tile) {
    ulonglong4 __1;
    uint4 v_ = *(uint4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    ((float2*)(&__1))[1] = __half22float2(((half2*)(&v_))[1]);
    ((float2*)(&__1))[2] = __half22float2(((half2*)(&v_))[2]);
    ((float2*)(&__1))[3] = __half22float2(((half2*)(&v_))[3]);
    ulonglong4 x_val = __1;
    ulonglong4 __2;
      ulonglong4 v__1 = *(ulonglong4*)(sum_local + 0);
      *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(x_val.x)));
      *(float2*)(&(__2.y)) = tl::add2(*(float2*)(&(v__1.y)), *(float2*)(&(x_val.y)));
      *(float2*)(&(__2.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(x_val.z)));
      *(float2*)(&(__2.w)) = tl::add2(*(float2*)(&(v__1.w)), *(float2*)(&(x_val.w)));
    *(ulonglong4*)(sum_local + 0) = __2;
    ulonglong4 __3;
      ulonglong4 v__2 = *(ulonglong4*)(sumsq_local + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(x_val.x)), *(float2*)(&(x_val.x)), *(float2*)(&(v__2.x)));
      *(float2*)(&(__3.y)) = tl::fma2(*(float2*)(&(x_val.y)), *(float2*)(&(x_val.y)), *(float2*)(&(v__2.y)));
      *(float2*)(&(__3.z)) = tl::fma2(*(float2*)(&(x_val.z)), *(float2*)(&(x_val.z)), *(float2*)(&(v__2.z)));
      *(float2*)(&(__3.w)) = tl::fma2(*(float2*)(&(x_val.w)), *(float2*)(&(x_val.w)), *(float2*)(&(v__2.w)));
    *(ulonglong4*)(sumsq_local + 0) = __3;
  }
  sum_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    sum_row[0] = (sum_row[0] + sum_local[rv]);
  }
  sum_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sum_row[0], (&(((float*)workspace_1)[0])));
  sumsq_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
    sumsq_row[0] = (sumsq_row[0] + sumsq_local[rv_1]);
  }
  __syncthreads();
  sumsq_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sumsq_row[0], (&(((float*)workspace)[0])));
  mean_row[0] = (sum_row[0] / 0x1.4p+13f/*1.024000e+04*/);
  float variance = ((sumsq_row[0] / 0x1.4p+13f/*1.024000e+04*/) - (mean_row[0] * mean_row[0]));
  rstd_row[0] = rsqrtf((variance + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  for (int tile_1 = 0; tile_1 < 10; ++tile_1) {
    *(uint4*)(X_local_cast + 0) = *(uint4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    *(uint4*)(weight_local_cast_1 + 0) = *(uint4*)(weight + ((tile_1 * 1024) + (((int)threadIdx.x) * 8)));
    *(uint4*)(bias_local_cast_2 + 0) = *(uint4*)(bias + ((tile_1 * 1024) + (((int)threadIdx.x) * 8)));
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      uint2 __4;
      float4 __5;
        float4 __6;
          float4 __7;
            float4 __8;
            uint2 v__3 = *(uint2*)(X_local_cast + (i_2 * 4));
            ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__3))[0]);
            ((float2*)(&__8))[1] = __half22float2(((half2*)(&v__3))[1]);
            float4 v__4 = make_float4(mean_row[0], mean_row[0], mean_row[0], mean_row[0]);
            *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__4.x)));
            *(float2*)(&(__7.z)) = tl::sub2(*(float2*)(&(__8.z)), *(float2*)(&(v__4.z)));
          float4 v__5 = make_float4(rstd_row[0], rstd_row[0], rstd_row[0], rstd_row[0]);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__5.x)));
          *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__5.z)));
        float4 __9;
        uint2 v__6 = *(uint2*)(weight_local_cast_1 + (i_2 * 4));
        ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__6))[0]);
        ((float2*)(&__9))[1] = __half22float2(((half2*)(&v__6))[1]);
        float4 __10;
        uint2 v__7 = *(uint2*)(bias_local_cast_2 + (i_2 * 4));
        ((float2*)(&__10))[0] = __half22float2(((half2*)(&v__7))[0]);
        ((float2*)(&__10))[1] = __half22float2(((half2*)(&v__7))[1]);
        *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(__6.x)), *(float2*)(&(__9.x)), *(float2*)(&(__10.x)));
        *(float2*)(&(__5.z)) = tl::fma2(*(float2*)(&(__6.z)), *(float2*)(&(__9.z)), *(float2*)(&(__10.z)));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
      *(uint2*)(Y_local + (i_2 * 4)) = __4;
    }
    *(uint4*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(Y_local + 0);
  }
}

