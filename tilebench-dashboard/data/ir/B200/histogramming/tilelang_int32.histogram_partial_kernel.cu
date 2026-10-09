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

extern "C" __global__ void histogram_partial_kernel_kernel(int* __restrict__ partial, const int* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) histogram_partial_kernel_kernel(int* __restrict__ partial, const int* __restrict__ x) {
  extern __shared__ __align__(1024) int smem[];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int broadcast_var = 0;
    *(int4*)(smem + ((i * 1024) + (((int)threadIdx.x) * 4))) = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  __syncthreads();
  for (int _tmp = 0; _tmp < 128; ++_tmp) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      int val = x[((((_tmp * 524288) + (((int)blockIdx.x) * 2048)) + (i_1 * 256)) + ((int)threadIdx.x))];
      if ((0 <= val) && (val < 4096)) {
        AtomicAdd((&(smem[val])), 1);
      }
    }
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(partial[(((int)blockIdx.x) * 4096)])), (&(smem[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

