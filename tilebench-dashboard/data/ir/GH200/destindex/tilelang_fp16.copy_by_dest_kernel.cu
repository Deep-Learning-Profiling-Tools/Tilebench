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

extern "C" __global__ void copy_by_dest_kernel_kernel(const int64_t* __restrict__ dest, const half_t* __restrict__ kv, half_t* __restrict__ out);
extern "C" __global__ void __launch_bounds__(64, 1) copy_by_dest_kernel_kernel(const int64_t* __restrict__ dest, const half_t* __restrict__ kv, half_t* __restrict__ out) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int dest_index = ((int)dest[(((((int)blockIdx.x) * 2) + (i >> 2)) / 3)]);
    if (0 <= dest_index) {
      if (dest_index < 40960) {
        *(uint1*)(out + (((dest_index * 1536) + ((((((int)blockIdx.x) * 8) + i) % 12) * 128)) + (((int)threadIdx.x) * 2))) = *(uint1*)(kv + (((((int)blockIdx.x) * 1024) + (i * 128)) + (((int)threadIdx.x) * 2)));
      }
    }
  }
}

