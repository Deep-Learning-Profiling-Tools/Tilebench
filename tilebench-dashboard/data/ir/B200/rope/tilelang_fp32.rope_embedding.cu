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

extern "C" __global__ void rope_embedding_kernel(float* __restrict__ Q, const float* __restrict__ cos, const float* __restrict__ sin);
extern "C" __global__ void __launch_bounds__(128, 1) rope_embedding_kernel(float* __restrict__ Q, const float* __restrict__ cos_1, const float* __restrict__ sin_1) {
  float cos_tile[4];
  float sin_tile[4];
  float q1_tile[8];
  float q2_tile[8];
  *(float4*)(cos_tile + 0) = *(float4*)(cos_1 + ((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) & 15) * 4)));
  *(float4*)(sin_tile + 0) = *(float4*)(sin_1 + ((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) & 15) * 4)));
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(q1_tile + (i * 4)) = *(float4*)(Q + (((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(q2_tile + (i_1 * 4)) = *(float4*)(Q + ((((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + 64));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    float q1 = q1_tile[i_2];
    float q2 = q2_tile[i_2];
    float c = cos_tile[(i_2 & 3)];
    float s = sin_tile[(i_2 & 3)];
    q1_tile[i_2] = ((q1 * c) - (q2 * s));
    q2_tile[i_2] = ((q2 * c) + (q1 * s));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(float4*)(Q + (((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4))) = *(float4*)(q1_tile + (i_3 * 4));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    *(float4*)(Q + ((((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 2048)) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + 64)) = *(float4*)(q2_tile + (i_4 * 4));
  }
}

