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

extern "C" __global__ void gaussian_blur_kernel_kernel(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) gaussian_blur_kernel_kernel(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output) {
  float acc[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int kr = 0; kr < 7; ++kr) {
    #pragma unroll
    for (int kc = 0; kc < 7; ++kc) {
      float w = kernel[((kr * 7) + kc)];
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        float condval;
        if (((((3 <= ((((((int)blockIdx.y) * 256) + ((i_1 & 1) * 128)) + ((int)threadIdx.x)) + kc)) && (((((((int)blockIdx.y) * 256) + ((i_1 & 1) * 128)) + ((int)threadIdx.x)) + kc) < 10243)) && (3 <= (((((int)blockIdx.x) * 4) + (i_1 >> 1)) + kr))) && ((((((int)blockIdx.x) * 4) + (i_1 >> 1)) + kr) < 10243))) {
          condval = input[((((((((((int)blockIdx.x) * 40960) + ((i_1 >> 1) * 10240)) + (kr * 10240)) + (((int)blockIdx.y) * 256)) + ((i_1 & 1) * 128)) + ((int)threadIdx.x)) + kc) - 30723)];
        } else {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        acc[i_1] = (acc[i_1] + (condval * w));
      }
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    output[(((((((int)blockIdx.x) * 40960) + ((i_2 >> 1) * 10240)) + (((int)blockIdx.y) * 256)) + ((i_2 & 1) * 128)) + ((int)threadIdx.x))] = acc[i_2];
  }
}

