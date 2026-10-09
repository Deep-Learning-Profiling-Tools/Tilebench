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

extern "C" __global__ void pad_kernel_kernel(const half_t* __restrict__ data, half_t* __restrict__ work);
extern "C" __global__ void __launch_bounds__(128, 1) pad_kernel_kernel(const half_t* __restrict__ data, half_t* __restrict__ work) {
  half_t broadcast_var = std::numeric_limits<half_t>::infinity();
  uint2 condval;
  if ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 78125)) {
    condval = *(uint2*)(data + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)));
  } else {
    condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  *(uint2*)(work + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))) = condval;
}

