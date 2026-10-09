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

extern "C" __global__ void leaky_relu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) leaky_relu_kernel_kernel(half_t* __restrict__ output, const half_t* __restrict__ x) {
  half_t x_reg[8];
  half_t output_reg[8];
  half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval;
  if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 390625)) {
    condval = *(uint4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  } else {
    condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  *(uint4*)(x_reg + 0) = condval;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    half_t value = x_reg[i];
    output_reg[i] = ((half_t(0x0p+0f/*0.000000e+00*/) < value) ? value : (half_t(0x1.47ae147ae147bp-7f/*1.000000e-02*/) * value));
  }
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 390625) {
    *(uint4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_reg + 0);
  }
}

