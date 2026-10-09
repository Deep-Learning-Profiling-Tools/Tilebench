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
  for (int i_s = 0; i_s < 8; ++i_s) {
    bool valid = (((((int)threadIdx.x) * 8) + i_s) < 77);
    int condval;
    if ((((((int)threadIdx.x) * 8) + i_s) < 77)) {
      condval = sums[((((int)threadIdx.x) * 8) + i_s)];
    } else {
      condval = 0;
    }
    src_buffer[i_s] = condval;
    original[i_s] = src_buffer[i_s];
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(int4*)(scan_smem + ((((int)threadIdx.x) * 8) + (i * 4))) = *(int4*)(src_buffer + (i * 4));
  }
  __syncthreads();
  tl::CumSum1D<128, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 1024);
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(int4*)(src_buffer + (i_1 * 4)) = *(int4*)(scan_smem + ((((int)threadIdx.x) * 8) + (i_1 * 4)));
  }
  for (int i_s_1 = 0; i_s_1 < 8; ++i_s_1) {
    if (((((int)threadIdx.x) * 8) + i_s_1) < 77) {
      sums[((((int)threadIdx.x) * 8) + i_s_1)] = (src_buffer[i_s_1] - original[i_s_1]);
    }
  }
}

