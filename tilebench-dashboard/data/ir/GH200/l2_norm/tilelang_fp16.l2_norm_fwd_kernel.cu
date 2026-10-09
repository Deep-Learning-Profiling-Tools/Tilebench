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

extern "C" __global__ void l2_norm_fwd_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, int M);
extern "C" __global__ void __launch_bounds__(256, 1) l2_norm_fwd_kernel_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y, int M) {
  float acc[4];
  half_t X_local_cast[4];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  float inv_norm[1];
  half_t X_local_cast_1[4];
  half_t Y_local[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(acc + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int tile = 0; tile < 10; ++tile) {
    *(uint2*)(X_local_cast + 0) = *(uint2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    float4 __1;
      float4 v_ = *(float4*)(acc + 0);
      float4 __2;
        float4 __3;
        uint2 v__1 = *(uint2*)(X_local_cast + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
        float4 __4;
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
        ((float2*)(&__4))[1] = __half22float2(((half2*)(&v__1))[1]);
        __2.x = (__3.x*__4.x);
        __2.y = (__3.y*__4.y);
        __2.z = (__3.z*__4.z);
        __2.w = (__3.w*__4.w);
      __1.x = (v_.x+__2.x);
      __1.y = (v_.y+__2.y);
      __1.z = (v_.z+__2.z);
      __1.w = (v_.w+__2.w);
    *(float4*)(acc + 0) = __1;
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    row_sum[0] = (row_sum[0] + acc[rv]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(row_sum[0], (&(workspace[0])));
  inv_norm[0] = rsqrtf((row_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 10; ++tile_1) {
    *(uint2*)(X_local_cast_1 + 0) = *(uint2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    uint2 __5;
    float4 __6;
      float4 __7;
      uint2 v__2 = *(uint2*)(X_local_cast_1 + 0);
      ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__2))[0]);
      ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__2))[1]);
      float4 v__3 = make_float4(inv_norm[0], inv_norm[0], inv_norm[0], inv_norm[0]);
      __6.x = (__7.x*v__3.x);
      __6.y = (__7.y*v__3.y);
      __6.z = (__7.z*v__3.z);
      __6.w = (__7.w*v__3.w);
    ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
    ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&__6))[1]);
    *(uint2*)(Y_local + 0) = __5;
    *(uint2*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(Y_local + 0);
  }
}

