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
extern "C" __global__ void __launch_bounds__(128, 1) matrix_transpose_kernel_kernel(__grid_constant__ const CUtensorMap output_desc, const half_t* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* tile_T = ((void*)((char*)buf_dyn_shmem + 32768));
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(output_desc);
  }
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(((half_t*)tile) + ((i * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(x + (((((((int)blockIdx.x) * 2621440) + (i * 163840)) + ((((int)threadIdx.x) >> 4) * 20480)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 128; ++i_1) {
    ((half_t*)tile_T)[(((((((i_1 >> 6) * 8192) + (((int)threadIdx.x) * 64)) + (((((i_1 & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_1 & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((i_1 & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (i_1 & 7))] = ((half_t*)tile)[((i_1 * 128) + ((int)threadIdx.x))];
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store(output_desc, (&(((half_t*)tile_T)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
    tl::tma_store(output_desc, (&(((half_t*)tile_T)[8192])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 128));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

