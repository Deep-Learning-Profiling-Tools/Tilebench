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

extern "C" __global__ void main_kernel(int* __restrict__ sums);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(int* __restrict__ sums) {
  int src_buffer[8];
  int original[8];
  extern __shared__ __align__(1024) int scan_smem[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    for (int vec_s = 0; vec_s < 4; ++vec_s) {
      bool valid = ((((i * 512) + (((int)threadIdx.x) * 4)) + vec_s) < 77);
      int condval;
      if (((((i * 512) + (((int)threadIdx.x) * 4)) + vec_s) < 77)) {
        condval = sums[(((i * 512) + (((int)threadIdx.x) * 4)) + vec_s)];
      } else {
        condval = 0;
      }
      src_buffer[((i * 4) + vec_s)] = condval;
      original[((i * 4) + vec_s)] = src_buffer[((i * 4) + vec_s)];
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(int4*)(scan_smem + ((i_1 * 512) + (((int)threadIdx.x) * 4))) = *(int4*)(src_buffer + (i_1 * 4));
  }
  __syncthreads();
  tl::CumSum1D<128, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 1024);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(int4*)(src_buffer + (i_2 * 4)) = *(int4*)(scan_smem + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    for (int vec_s_1 = 0; vec_s_1 < 4; ++vec_s_1) {
      if ((((i_3 * 512) + (((int)threadIdx.x) * 4)) + vec_s_1) < 77) {
        sums[(((i_3 * 512) + (((int)threadIdx.x) * 4)) + vec_s_1)] = (src_buffer[((i_3 * 4) + vec_s_1)] - original[((i_3 * 4) + vec_s_1)]);
      }
    }
  }
}

