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
extern "C" __global__ void __launch_bounds__(256, 1) apply_batch_norm_kernel_kernel(const float* __restrict__ beta, const float* __restrict__ gamma, const float* __restrict__ input, const float* __restrict__ inv_std, const float* __restrict__ mean, float* __restrict__ output) {
  float mean_local[4];
  float inv_std_local[4];
  float gamma_local[4];
  float beta_local[4];
  float scale[4];
  float shift[4];
  float x[32];
  float y[32];
  *(float4*)(mean_local + 0) = *(float4*)(mean + (((int)threadIdx.x) * 4));
  *(float4*)(inv_std_local + 0) = *(float4*)(inv_std + (((int)threadIdx.x) * 4));
  *(float4*)(gamma_local + 0) = *(float4*)(gamma + (((int)threadIdx.x) * 4));
  *(float4*)(beta_local + 0) = *(float4*)(beta + (((int)threadIdx.x) * 4));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    scale[i] = (inv_std_local[i] * gamma_local[i]);
    shift[i] = (beta_local[i] - (mean_local[i] * scale[i]));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(float4*)(x + (i_1 * 4)) = *(float4*)(input + (((((int)blockIdx.x) * 8192) + (i_1 * 1024)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    y[i_2] = ((x[i_2] * scale[(i_2 & 3)]) + shift[(i_2 & 3)]);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(float4*)(output + (((((int)blockIdx.x) * 8192) + (i_3 * 1024)) + (((int)threadIdx.x) * 4))) = *(float4*)(y + (i_3 * 4));
  }
}

