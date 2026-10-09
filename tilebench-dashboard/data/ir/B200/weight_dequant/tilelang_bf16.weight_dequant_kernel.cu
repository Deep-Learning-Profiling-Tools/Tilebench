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
  bfloat16_t X_local_cast_1[8];
  bfloat16_t S_local_cast_2[8];
  bfloat16_t output_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(X_local_cast_1 + 0) = *(uint4*)(X + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    *(uint4*)(S_local_cast_2 + 0) = make_uint4(__pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))]), __pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))]), __pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))]), __pack_nv_bfloat162(S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))], S[(((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (i * 8)) + (((int)threadIdx.x) >> 4))]));
    for (int vec = 0; vec < 2; ++vec) {
      uint2 __1;
      float4 __2;
        float4 __3;
        uint2 v_ = *(uint2*)(X_local_cast_1 + (vec * 4));
        ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
        float4 __4;
        uint2 v__1 = *(uint2*)(S_local_cast_2 + (vec * 4));
        ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
        *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(__4.z)));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
      *(uint2*)(output_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(output + (((((int)blockIdx.x) * 2048) + (i * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_local_cast + 0);
  }
}

