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

extern "C" __global__ void main_kernel(const int* __restrict__ chunk_offsets, int* __restrict__ src);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ chunk_offsets, int* __restrict__ src) {
  int src_buffer[8];
  int original[8];
  extern __shared__ __align__(1024) int scan_smem[];
  int base = chunk_offsets[((int)blockIdx.x)];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int broadcast_var = 0;
    int4 condval;
    if (((((((int)blockIdx.x) * 64) + (i * 32)) + (((int)threadIdx.x) >> 2)) < 4883)) {
      condval = *(int4*)(src + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    } else {
      condval = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    *(int4*)(src_buffer + (i * 4)) = condval;
    *(int4*)(original + (i * 4)) = *(int4*)(src_buffer + (i * 4));
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
    if ((((((int)blockIdx.x) * 64) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4883) {
      int4 __1;
        int4 __2;
          int4 v_ = *(int4*)(src_buffer + (i_3 * 4));
          int4 v__1 = make_int4(base, base, base, base);
          __2.x = (v_.x+v__1.x);
          __2.y = (v_.y+v__1.y);
          __2.z = (v_.z+v__1.z);
          __2.w = (v_.w+v__1.w);
        int4 v__2 = *(int4*)(original + (i_3 * 4));
        __1.x = (__2.x-v__2.x);
        __1.y = (__2.y-v__2.y);
        __1.z = (__2.z-v__2.z);
        __1.w = (__2.w-v__2.w);
      *(int4*)(src + (((((int)blockIdx.x) * 1024) + (i_3 * 512)) + (((int)threadIdx.x) * 4))) = __1;
    }
  }
}

