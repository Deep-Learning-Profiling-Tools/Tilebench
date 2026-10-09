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
  int broadcast_var = 0;
  longlong4 condval;
  if ((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 4883)) {
    condval = tl::load_global_256(&(*(longlong4*)(src + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)))));
  } else {
    condval = make_longlong4(((long long)(int)(broadcast_var)) | ((long long)(int)(broadcast_var) << 32), ((long long)(int)(broadcast_var)) | ((long long)(int)(broadcast_var) << 32), ((long long)(int)(broadcast_var)) | ((long long)(int)(broadcast_var) << 32), ((long long)(int)(broadcast_var)) | ((long long)(int)(broadcast_var) << 32));
  }
  *(longlong4*)(src_buffer + 0) = condval;
  *(longlong4*)(original + 0) = *(longlong4*)(src_buffer + 0);
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
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 4883) {
    longlong4 __1;
      longlong4 __2;
        longlong4 v_ = *(longlong4*)(src_buffer + 0);
        longlong4 v__1 = make_longlong4(((long long)(int)(base)) | ((long long)(int)(base) << 32), ((long long)(int)(base)) | ((long long)(int)(base) << 32), ((long long)(int)(base)) | ((long long)(int)(base) << 32), ((long long)(int)(base)) | ((long long)(int)(base) << 32));
        ((int2*)(&(__2.x)))->x = (((int2*)(&(v_.x)))->x+((int2*)(&(v__1.x)))->x);
        ((int2*)(&(__2.x)))->y = (((int2*)(&(v_.x)))->y+((int2*)(&(v__1.x)))->y);
        ((int2*)(&(__2.y)))->x = (((int2*)(&(v_.y)))->x+((int2*)(&(v__1.y)))->x);
        ((int2*)(&(__2.y)))->y = (((int2*)(&(v_.y)))->y+((int2*)(&(v__1.y)))->y);
        ((int2*)(&(__2.z)))->x = (((int2*)(&(v_.z)))->x+((int2*)(&(v__1.z)))->x);
        ((int2*)(&(__2.z)))->y = (((int2*)(&(v_.z)))->y+((int2*)(&(v__1.z)))->y);
        ((int2*)(&(__2.w)))->x = (((int2*)(&(v_.w)))->x+((int2*)(&(v__1.w)))->x);
        ((int2*)(&(__2.w)))->y = (((int2*)(&(v_.w)))->y+((int2*)(&(v__1.w)))->y);
      longlong4 v__2 = *(longlong4*)(original + 0);
      ((int2*)(&(__1.x)))->x = (((int2*)(&(__2.x)))->x-((int2*)(&(v__2.x)))->x);
      ((int2*)(&(__1.x)))->y = (((int2*)(&(__2.x)))->y-((int2*)(&(v__2.x)))->y);
      ((int2*)(&(__1.y)))->x = (((int2*)(&(__2.y)))->x-((int2*)(&(v__2.y)))->x);
      ((int2*)(&(__1.y)))->y = (((int2*)(&(__2.y)))->y-((int2*)(&(v__2.y)))->y);
      ((int2*)(&(__1.z)))->x = (((int2*)(&(__2.z)))->x-((int2*)(&(v__2.z)))->x);
      ((int2*)(&(__1.z)))->y = (((int2*)(&(__2.z)))->y-((int2*)(&(v__2.z)))->y);
      ((int2*)(&(__1.w)))->x = (((int2*)(&(__2.w)))->x-((int2*)(&(v__2.w)))->x);
      ((int2*)(&(__1.w)))->y = (((int2*)(&(__2.w)))->y-((int2*)(&(v__2.w)))->y);
    tl::store_global_256(&(*(longlong4*)(src + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)))), __1);
  }
}

