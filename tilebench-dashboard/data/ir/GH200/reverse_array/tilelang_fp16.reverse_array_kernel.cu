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

extern "C" __global__ void reverse_array_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) reverse_array_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output) {
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (((((int)blockIdx.x) * 8) + (i >> 1)) < 78125) {
      half_t condval;
      if ((-19999999 <= (((0 - ((int)threadIdx.x)) - (i * 128)) - (((int)blockIdx.x) * 2048)))) {
        condval = input[(((19999999 - ((int)threadIdx.x)) - (i * 128)) - (((int)blockIdx.x) * 2048))];
      } else {
        condval = half_t(0x0p+0f/*0.000000e+00*/);
      }
      output[(((((int)blockIdx.x) * 2048) + (i * 128)) + ((int)threadIdx.x))] = condval;
    }
  }
}

