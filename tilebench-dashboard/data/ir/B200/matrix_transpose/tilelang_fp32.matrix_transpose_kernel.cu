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

extern "C" __global__ void matrix_transpose_kernel_kernel(__grid_constant__ const CUtensorMap output_desc, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) matrix_transpose_kernel_kernel(__grid_constant__ const CUtensorMap output_desc, const float* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* tile_T = ((void*)((char*)buf_dyn_shmem + 4096));
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(output_desc);
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(((float*)tile) + ((i * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(x + (((((((int)blockIdx.x) * 655360) + (i * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    ((float*)tile_T)[((((((((int)threadIdx.x) & 31) * 32) + ((((i_1 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((i_1 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((i_1 & 1) + (((int)threadIdx.x) & 1)) & 1) * 4)) + (((int)threadIdx.x) >> 5))] = ((float*)tile)[((i_1 * 128) + ((int)threadIdx.x))];
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store(output_desc, (&(((float*)tile_T)[0])), (((int)blockIdx.x) * 32), (((int)blockIdx.y) * 32));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

