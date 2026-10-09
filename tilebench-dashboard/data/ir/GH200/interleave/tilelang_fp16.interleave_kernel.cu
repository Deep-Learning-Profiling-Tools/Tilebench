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

extern "C" __global__ void interleave_kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(256, 1) interleave_kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ output) {
  half_t A_reg[4];
  half_t B_reg[4];
  half_t output_reg[8];
  half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
  uint2 condval;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)) < 78125)) {
    condval = *(uint2*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  } else {
    condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  *(uint2*)(A_reg + 0) = condval;
  half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
  uint2 condval_1;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)) < 78125)) {
    condval_1 = *(uint2*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  } else {
    condval_1 = make_uint2(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
  }
  *(uint2*)(B_reg + 0) = condval_1;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    output_reg[i] = (((i % 2) == 0) ? A_reg[(i >> 1)] : B_reg[(i >> 1)]);
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)) < 78125) {
    *(uint4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(output_reg + 0);
  }
}

