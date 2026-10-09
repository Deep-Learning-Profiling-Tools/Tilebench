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

extern "C" __global__ void histogram_reduce_kernel_kernel(int* __restrict__ histogram, const int* __restrict__ partial);
extern "C" __global__ void __launch_bounds__(128, 1) histogram_reduce_kernel_kernel(int* __restrict__ histogram, const int* __restrict__ partial) {
  int acc[4];
  int tile[64];
  int tile_sum[4];
  extern __shared__ __align__(1024) int workspace[];
  int broadcast_var = 0;
  *(int4*)(acc + 0) = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int _tmp = 0; _tmp < 2; ++_tmp) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      int broadcast_var_1 = 0;
      *(int4*)(tile + (i * 4)) = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(int4*)(tile + (i_1 * 4)) = *(int4*)(partial + (((((_tmp * 524288) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      tile_sum[i_2] = 0;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        tile_sum[i_2] = (tile_sum[i_2] + tile[((rv * 4) + i_2)]);
      }
      tile_sum[i_2] = tl::AllReduce<tl::SumOp, 128, 16, 0, tl::NamedBarrier<128>>::run(tile_sum[i_2], (&(workspace[0])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      acc[i_3] = (acc[i_3] + tile_sum[i_3]);
    }
  }
  if ((((int)threadIdx.x) >> 4) == 0) {
    *(int4*)(histogram + ((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) & 15) * 4))) = *(int4*)(acc + 0);
  }
}

