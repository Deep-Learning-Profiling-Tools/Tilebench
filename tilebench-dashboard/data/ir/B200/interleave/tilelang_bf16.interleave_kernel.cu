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

extern "C" __global__ void interleave_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) interleave_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ output) {
  bfloat16_t A_reg[8];
  bfloat16_t B_reg[8];
  bfloat16_t output_reg[16];
  bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 78125)) {
    condval = *(uint4*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
  } else {
    condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
  }
  *(uint4*)(A_reg + 0) = condval;
  bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval_1;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 78125)) {
    condval_1 = *(uint4*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
  } else {
    condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
  }
  *(uint4*)(B_reg + 0) = condval_1;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    output_reg[i] = (((i % 2) == 0) ? A_reg[(i >> 1)] : B_reg[(i >> 1)]);
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 78125) {
    tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 16)))), *(ulonglong4*)(output_reg + 0));
  }
}

