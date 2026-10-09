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
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  ulonglong4 condval;
  if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < 625)) {
    condval = tl::load_global_256(&(*(ulonglong4*)(X + ((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))));
  } else {
    condval = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var));
  }
  *(ulonglong4*)(x_frag + 0) = condval;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    y_frag[i] = ((0x0p+0f/*0.000000e+00*/ < x_frag[i]) ? (x_frag[i] + 0x1p+0f/*1.000000e+00*/) : expf(x_frag[i]));
  }
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < 625) {
    tl::store_global_256(&(*(ulonglong4*)(Y + ((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))), *(ulonglong4*)(y_frag + 0));
  }
}

