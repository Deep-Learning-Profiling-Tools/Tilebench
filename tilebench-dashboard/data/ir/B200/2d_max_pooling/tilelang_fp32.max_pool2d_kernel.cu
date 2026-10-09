#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void max_pool2d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) max_pool2d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output) {
  float acc[4];
  float neg_inf = -CUDART_INF_F;
  float broadcast_var = -CUDART_INF_F;
  *(float4*)(acc + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  #pragma unroll
  for (int kh = 0; kh < 3; ++kh) {
    #pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float condval;
        if ((((1 <= (((i * 256) + (((int)threadIdx.x) * 2)) + kw)) && ((((i * 256) + (((int)threadIdx.x) * 2)) + kw) < 641)) && (1 <= ((((int)blockIdx.y) * 2) + kh)))) {
          condval = input[(((((((((int)blockIdx.x) * 409600) + (((int)blockIdx.y) * 1280)) + (kh * 640)) + (i * 256)) + (((int)threadIdx.x) * 2)) + kw) - 641)];
        } else {
          condval = -CUDART_INF_F;
        }
        acc[i] = max(acc[i], condval);
      }
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    if (((i_1 * 2) + (((int)threadIdx.x) >> 6)) < 5) {
      output[((((((int)blockIdx.x) * 102400) + (((int)blockIdx.y) * 320)) + (i_1 * 128)) + ((int)threadIdx.x))] = acc[i_1];
    }
  }
}

