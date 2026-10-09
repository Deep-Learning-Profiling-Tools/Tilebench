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

extern "C" __global__ void weight_dequant_kernel_kernel(const float* __restrict__ S, const float* __restrict__ X, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(256, 1) weight_dequant_kernel_kernel(const float* __restrict__ S, const float* __restrict__ X, float* __restrict__ output) {
  ulonglong4 __1;
    ulonglong4 v_ = tl::load_global_256(&(*(ulonglong4*)(X + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
    ulonglong4 v__1 = make_ulonglong4(*(unsigned long long*)&make_float2(S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))], S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))]), *(unsigned long long*)&make_float2(S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))], S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))]), *(unsigned long long*)&make_float2(S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))], S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))]), *(unsigned long long*)&make_float2(S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))], S[((((((int)blockIdx.x) / 640) * 80) + ((((int)blockIdx.x) % 5) * 16)) + (((int)threadIdx.x) >> 4))]));
    *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(&(__1.y)) = tl::mul2(*(float2*)(&(v_.y)), *(float2*)(&(v__1.y)));
    *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
    *(float2*)(&(__1.w)) = tl::mul2(*(float2*)(&(v_.w)), *(float2*)(&(v__1.w)));
  ulonglong4 value = __1;
  tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))), value);
}

