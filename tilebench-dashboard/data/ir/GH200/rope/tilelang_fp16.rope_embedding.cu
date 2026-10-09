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

extern "C" __global__ void rope_embedding_kernel(half_t* __restrict__ Q, const half_t* __restrict__ cos, const half_t* __restrict__ sin);
extern "C" __global__ void __launch_bounds__(256, 1) rope_embedding_kernel(half_t* __restrict__ Q, const half_t* __restrict__ cos_1, const half_t* __restrict__ sin_1) {
  half_t cos_tile[4];
  half_t sin_tile[4];
  half_t q1_tile[4];
  half_t q2_tile[4];
  *(uint2*)(cos_tile + 0) = *(uint2*)(cos_1 + ((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) & 15) * 4)));
  *(uint2*)(sin_tile + 0) = *(uint2*)(sin_1 + ((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) & 15) * 4)));
  *(uint2*)(q1_tile + 0) = *(uint2*)(Q + ((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)));
  *(uint2*)(q2_tile + 0) = *(uint2*)(Q + (((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + 64));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    half_t q1 = q1_tile[i];
    half_t q2 = q2_tile[i];
    half_t c = cos_tile[i];
    half_t s = sin_tile[i];
    q1_tile[i] = ((q1 * c) - (q2 * s));
    q2_tile[i] = ((q2 * c) + (q1 * s));
  }
  *(uint2*)(Q + ((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4))) = *(uint2*)(q1_tile + 0);
  *(uint2*)(Q + (((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + 64)) = *(uint2*)(q2_tile + 0);
}

