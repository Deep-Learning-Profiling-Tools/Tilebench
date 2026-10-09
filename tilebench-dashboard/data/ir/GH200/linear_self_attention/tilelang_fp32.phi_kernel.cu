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

extern "C" __global__ void phi_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) phi_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y) {
  float x_frag[8];
  float y_frag[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    float4 condval;
    if ((((((int)blockIdx.x) * 2) + i) < 625)) {
      condval = *(float4*)(X + (((((((int)blockIdx.x) * 8192) + (i * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    } else {
      condval = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    *(float4*)(x_frag + (i * 4)) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    y_frag[i_1] = ((0x0p+0f/*0.000000e+00*/ < x_frag[i_1]) ? (x_frag[i_1] + 0x1p+0f/*1.000000e+00*/) : expf(x_frag[i_1]));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    if (((((int)blockIdx.x) * 2) + i_2) < 625) {
      *(float4*)(Y + (((((((int)blockIdx.x) * 8192) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4))) = *(float4*)(y_frag + (i_2 * 4));
    }
  }
}

