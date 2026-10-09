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

extern "C" __global__ void apply_batch_norm_kernel_kernel(const float* __restrict__ beta, const float* __restrict__ gamma, const float* __restrict__ input, const float* __restrict__ inv_std, const float* __restrict__ mean, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) apply_batch_norm_kernel_kernel(const float* __restrict__ beta, const float* __restrict__ gamma, const float* __restrict__ input, const float* __restrict__ inv_std, const float* __restrict__ mean, float* __restrict__ output) {
  float mean_local[8];
  float inv_std_local[8];
  float gamma_local[8];
  float beta_local[8];
  float scale[8];
  float shift[8];
  float x[64];
  float y[64];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(mean_local + (i * 4)) = *(float4*)(mean + ((i * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(inv_std_local + (i_1 * 4)) = *(float4*)(inv_std + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(float4*)(gamma_local + (i_2 * 4)) = *(float4*)(gamma + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(float4*)(beta_local + (i_3 * 4)) = *(float4*)(beta + ((i_3 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    scale[i_4] = (inv_std_local[i_4] * gamma_local[i_4]);
    shift[i_4] = (beta_local[i_4] - (mean_local[i_4] * scale[i_4]));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 16; ++i_5) {
    *(float4*)(x + (i_5 * 4)) = *(float4*)(input + (((((int)blockIdx.x) * 8192) + (i_5 * 512)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 64; ++i_6) {
    y[i_6] = ((x[i_6] * scale[(i_6 & 7)]) + shift[(i_6 & 7)]);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    *(float4*)(output + (((((int)blockIdx.x) * 8192) + (i_7 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(y + (i_7 * 4));
  }
}

