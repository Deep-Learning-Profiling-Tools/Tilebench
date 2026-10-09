#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void argmax_rowwise_kernel_kernel(int64_t* __restrict__ Out, const float* __restrict__ X, int M);
extern "C" __global__ void __launch_bounds__(128, 1) argmax_rowwise_kernel_kernel(int64_t* __restrict__ Out, const float* __restrict__ X, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 512));
  float best_val[1];
  int best_idx[1];
  float x_tile[16];
  float tile_max[1];
  int idx_tile[16];
  int tile_idx[1];
  best_val[0] = -CUDART_INF_F;
  best_idx[0] = 0;
  __syncthreads();
  for (int tile = 0; tile < 10; ++tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = -CUDART_INF_F;
      *(float4*)(x_tile + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(ulonglong4*)(x_tile + (i_1 * 8)) = tl::load_global_256(&(*(ulonglong4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)tile) * (int64_t)2048)) + (((int64_t)i_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
    }
    tile_max[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      tile_max[0] = max(tile_max[0], x_tile[(((rv & 1) * 8) + (rv >> 1))]);
    }
    tile_max[0] = tl::AllReduce<tl::MaxOp, 128, 1, 0, tl::NamedBarrier<128>>::run(tile_max[0], (&(((float*)workspace_1)[0])));
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      idx_tile[i_2] = ((x_tile[i_2] == tile_max[0]) ? ((((tile * 2048) + ((i_2 >> 3) * 1024)) + (((int)threadIdx.x) * 8)) + (i_2 & 7)) : 20480);
    }
    tile_idx[0] = 2147483647;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      tile_idx[0] = min(tile_idx[0], idx_tile[(((rv_1 & 1) * 8) + (rv_1 >> 1))]);
    }
    tile_idx[0] = tl::AllReduce<tl::MinOp, 128, 1, 0, tl::NamedBarrier<128>>::run(tile_idx[0], (&(((int*)workspace)[0])));
    bool better = (best_val[0] < tile_max[0]);
    best_val[0] = (better ? tile_max[0] : best_val[0]);
    best_idx[0] = (better ? tile_idx[0] : best_idx[0]);
  }
  Out[((int64_t)((int)blockIdx.x))] = ((int64_t)best_idx[0]);
}

