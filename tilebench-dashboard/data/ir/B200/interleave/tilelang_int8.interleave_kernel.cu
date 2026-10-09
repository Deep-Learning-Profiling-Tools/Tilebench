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

extern "C" __global__ void interleave_kernel_kernel(const signed char* __restrict__ A, const signed char* __restrict__ B, signed char* __restrict__ output);
extern "C" __global__ void __launch_bounds__(256, 1) interleave_kernel_kernel(const signed char* __restrict__ A, const signed char* __restrict__ B, signed char* __restrict__ output) {
  signed char A_reg[16];
  signed char B_reg[16];
  signed char output_reg[32];
  signed char broadcast_var = (signed char)0;
  int4 condval;
  if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 78125)) {
    condval = *(int4*)(A + ((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 16)));
  } else {
    condval = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(int4*)(A_reg + 0) = condval;
  signed char broadcast_var_1 = (signed char)0;
  int4 condval_1;
  if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 78125)) {
    condval_1 = *(int4*)(B + ((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 16)));
  } else {
    condval_1 = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  *(int4*)(B_reg + 0) = condval_1;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    output_reg[i] = (((i % 2) == 0) ? A_reg[(i >> 1)] : B_reg[(i >> 1)]);
  }
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 4)) < 78125) {
    tl::store_global_256(&(*(longlong4*)(output + ((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 32)))), *(longlong4*)(output_reg + 0));
  }
}

