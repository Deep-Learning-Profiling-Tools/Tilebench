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

extern "C" __global__ void matrix_transpose_kernel_kernel(__grid_constant__ const CUtensorMap output_desc, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) matrix_transpose_kernel_kernel(__grid_constant__ const CUtensorMap output_desc, const half_t* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* tile_T = ((void*)((char*)buf_dyn_shmem + 8192));
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(output_desc);
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(((half_t*)tile) + ((i * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(x + (((((((int)blockIdx.x) * 1310720) + (i * 655360)) + ((((int)threadIdx.x) >> 3) * 20480)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    ((half_t*)tile_T)[(((((((((int)threadIdx.x) & 63) * 64) + ((((i_1 >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_1 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((i_1 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((i_1 & 1) * 4)) + (((int)threadIdx.x) >> 6))] = ((half_t*)tile)[((i_1 * 256) + ((int)threadIdx.x))];
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(output_desc, (&(((half_t*)tile_T)[0])), (((int)blockIdx.x) * 64), (((int)blockIdx.y) * 64));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

