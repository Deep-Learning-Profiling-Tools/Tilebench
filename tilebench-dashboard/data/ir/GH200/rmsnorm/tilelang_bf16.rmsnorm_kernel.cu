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

extern "C" __global__ void rmsnorm_kernel_kernel(const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y, const bfloat16_t* __restrict__ rms_w, int M);
extern "C" __global__ void __launch_bounds__(256, 1) rmsnorm_kernel_kernel(const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y, const bfloat16_t* __restrict__ rms_w, int M) {
  float sumsq_local[4];
  bfloat16_t X_local_cast[4];
  float row_sum[1];
  extern __shared__ __align__(1024) float workspace[];
  float inv_rms[1];
  bfloat16_t X_local_cast_2[4];
  bfloat16_t rms_w_local_cast_3[4];
  bfloat16_t Y_local_cast_1[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(sumsq_local + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int tile = 0; tile < 10; ++tile) {
    *(uint2*)(X_local_cast + 0) = *(uint2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    float4 __1;
      float4 v_ = *(float4*)(sumsq_local + 0);
      float4 __2;
        float4 __3;
        uint2 v__1 = *(uint2*)(X_local_cast + 0);
        ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        float4 __4;
        ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        __2.x = (__3.x*__4.x);
        __2.y = (__3.y*__4.y);
        __2.z = (__3.z*__4.z);
        __2.w = (__3.w*__4.w);
      __1.x = (v_.x+__2.x);
      __1.y = (v_.y+__2.y);
      __1.z = (v_.z+__2.z);
      __1.w = (v_.w+__2.w);
    *(float4*)(sumsq_local + 0) = __1;
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    row_sum[0] = (row_sum[0] + sumsq_local[rv]);
  }
  row_sum[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(row_sum[0], (&(workspace[0])));
  inv_rms[0] = rsqrtf(((row_sum[0] / 0x1.4p+13f/*1.024000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 10; ++tile_1) {
    *(uint2*)(X_local_cast_2 + 0) = *(uint2*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    *(uint2*)(rms_w_local_cast_3 + 0) = *(uint2*)(rms_w + ((tile_1 * 1024) + (((int)threadIdx.x) * 4)));
    uint2 __5;
    float4 __6;
      float4 __7;
        float4 __8;
        uint2 v__2 = *(uint2*)(X_local_cast_2 + 0);
        ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
        ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
        float4 v__3 = make_float4(inv_rms[0], inv_rms[0], inv_rms[0], inv_rms[0]);
        __7.x = (__8.x*v__3.x);
        __7.y = (__8.y*v__3.y);
        __7.z = (__8.z*v__3.z);
        __7.w = (__8.w*v__3.w);
      float4 __9;
      uint2 v__4 = *(uint2*)(rms_w_local_cast_3 + 0);
      ((float2*)(&__9))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
      ((float2*)(&__9))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
      __6.x = (__7.x*__9.x);
      __6.y = (__7.y*__9.y);
      __6.z = (__7.z*__9.z);
      __6.w = (__7.w*__9.w);
    (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&__6))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&__6))[1]);
    *(uint2*)(Y_local_cast_1 + 0) = __5;
    *(uint2*)(Y + (((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(Y_local_cast_1 + 0);
  }
}

