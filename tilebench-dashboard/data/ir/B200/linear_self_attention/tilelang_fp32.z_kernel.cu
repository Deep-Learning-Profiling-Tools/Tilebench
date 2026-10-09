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

extern "C" __global__ void z_kernel_kernel(const float* __restrict__ PhiK, float* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(128, 1) z_kernel_kernel(const float* __restrict__ PhiK, float* __restrict__ Z) {
  float acc[8];
  float k_frag[8];
  float tile_sum[8];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int _tmp = 0; _tmp < 313; ++_tmp) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    ulonglong4 condval;
    if ((((_tmp * 2) + (((int)threadIdx.x) >> 6)) < 625)) {
      condval = tl::load_global_256(&(*(ulonglong4*)(PhiK + ((((_tmp * 8192) + ((((int)threadIdx.x) >> 2) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 3) * 8)))));
    } else {
      condval = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1), *(unsigned long long*)&make_float2(broadcast_var_1, broadcast_var_1));
    }
    *(ulonglong4*)(k_frag + 0) = condval;
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      tile_sum[i_1] = 0x0p+0f/*0.000000e+00*/;
      tile_sum[i_1] = (tile_sum[i_1] + k_frag[i_1]);
      tile_sum[i_1] = tl::AllReduce<tl::SumOp, 128, 4, 0, tl::NamedBarrier<128>>::run(tile_sum[i_1], (&(workspace[0])));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      acc[i_2] = (acc[i_2] + tile_sum[i_2]);
    }
  }
  if ((((int)threadIdx.x) >> 2) == 0) {
    tl::store_global_256(&(*(ulonglong4*)(Z + ((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) & 3) * 8)))), *(ulonglong4*)(acc + 0));
  }
}

