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

extern "C" __global__ void weight_dequant_kernel_kernel(const bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) weight_dequant_kernel_kernel(const bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ output) {
  bfloat16_t X_local_cast_1[4];
  bfloat16_t S_local_cast_2[4];
  bfloat16_t output_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint2*)(X_local_cast_1 + 0) = *(uint2*)(X + (((((int)blockIdx.x) * 2048) + (i * 512)) + (((int)threadIdx.x) * 4)));
    *(uint2*)(S_local_cast_2 + 0) = make_uint2(__pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 4)) + (((int)threadIdx.x) >> 5))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 4)) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 4)) + (((int)threadIdx.x) >> 5))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 4)) + (((int)threadIdx.x) >> 5))]));
    uint2 __1;
    float4 __2;
      float4 __3;
      uint2 v_ = *(uint2*)(X_local_cast_1 + 0);
      ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
      float4 __4;
      uint2 v__1 = *(uint2*)(S_local_cast_2 + 0);
      ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
      ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
      __2.x = (__3.x*__4.x);
      __2.y = (__3.y*__4.y);
      __2.z = (__3.z*__4.z);
      __2.w = (__3.w*__4.w);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
    *(uint2*)(output_local_cast + 0) = __1;
    *(uint2*)(output + (((((int)blockIdx.x) * 2048) + (i * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(output_local_cast + 0);
  }
}

