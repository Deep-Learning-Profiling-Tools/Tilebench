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

extern "C" __global__ void leaky_relu_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) leaky_relu_kernel_kernel(bfloat16_t* __restrict__ output, const bfloat16_t* __restrict__ x) {
  bfloat16_t x_reg[64];
  bfloat16_t output_reg[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((int)blockIdx.x) * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) < 390625)) {
      condval = *(uint4*)(x + (((((int)blockIdx.x) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(x_reg + (i * 8)) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    bfloat16_t value = x_reg[i_1];
    output_reg[i_1] = ((bfloat16_t(0x0p+0f/*0.000000e+00*/) < value) ? value : (bfloat16_t(0x1.47ae147ae147bp-7f/*1.000000e-02*/) * value));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    if ((((((int)blockIdx.x) * 64) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) < 390625) {
      *(uint4*)(output + (((((int)blockIdx.x) * 8192) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_reg + (i_2 * 8));
    }
  }
}

