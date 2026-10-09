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

extern "C" __global__ void rmsnorm_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, const half_t* __restrict__ rms_w, int M);
extern "C" __global__ void __launch_bounds__(256, 1) rmsnorm_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, const half_t* __restrict__ rms_w, int M) {
  float sumsq_local[8];
  half_t X_local_cast[8];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  float inv_rms[1];
  half_t X_local_cast_2[8];
  half_t rms_w_local_cast_3[8];
  half_t Y_local_cast_1[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int tile = 0; tile < 5; ++tile) {
    *(uint4*)(X_local_cast + 0) = *(uint4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(X_local_cast + (i_1 * 4));
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 __3;
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 v__1 = *(float4*)(sumsq_local + (i_1 * 4));
        *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(__2.x)), *(float2*)(&(__3.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__1.z)) = tl::fma2(*(float2*)(&(__2.z)), *(float2*)(&(__3.z)), *(float2*)(&(v__1.z)));
      *(float4*)(sumsq_local + (i_1 * 4)) = __1;
    }
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    row_sum[0] = (row_sum[0] + sumsq_local[rv]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(row_sum[0], (&(workspace[0])));
  inv_rms[0] = rsqrtf(((row_sum[0] / 0x1.4p+13f/*1.024000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 5; ++tile_1) {
    *(uint4*)(X_local_cast_2 + 0) = *(uint4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    *(uint4*)(rms_w_local_cast_3 + 0) = *(uint4*)(rms_w + ((tile_1 * 2048) + (((int)threadIdx.x) * 8)));
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      uint2 __4;
      float4 __5;
        float4 __6;
          float4 __7;
          uint2 v__2 = *(uint2*)(X_local_cast_2 + (i_2 * 4));
          ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__2))[0]);
          ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__2))[1]);
          float4 v__3 = make_float4(inv_rms[0], inv_rms[0], inv_rms[0], inv_rms[0]);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__3.z)));
        float4 __8;
        uint2 v__4 = *(uint2*)(rms_w_local_cast_3 + (i_2 * 4));
        ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__4))[0]);
        ((float2*)(&__8))[1] = __half22float2(((half2*)(&v__4))[1]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(__8.x)));
        *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(__6.z)), *(float2*)(&(__8.z)));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
      *(uint2*)(Y_local_cast_1 + (i_2 * 4)) = __4;
    }
    *(uint4*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(Y_local_cast_1 + 0);
  }
}

