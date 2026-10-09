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

extern "C" __global__ void compute_block_sums_kernel_kernel(float* __restrict__ block_sq_sum, float* __restrict__ block_sum, const float* __restrict__ input);
extern "C" __global__ void __launch_bounds__(128, 1) compute_block_sums_kernel_kernel(float* __restrict__ block_sq_sum, float* __restrict__ block_sum, const float* __restrict__ input) {
  float acc_sum[8];
  float acc_sq[8];
  float input_local[64];
  float input_sq_local[64];
  float tile_sum[8];
  float tile_sq_sum[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_sum + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_sq + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  #pragma unroll
  for (int _tmp = 0; _tmp < 8; ++_tmp) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      float4 condval;
      if ((((((int)blockIdx.x) * 2) + (_tmp >> 2)) < 625)) {
        condval = *(float4*)(input + ((((((int)blockIdx.x) * 65536) + (_tmp * 8192)) + (i_2 * 512)) + (((int)threadIdx.x) * 4)));
      } else {
        condval = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      *(float4*)(input_local + (i_2 * 4)) = condval;
      float4 __1;
        float4 v_ = *(float4*)(input_local + (i_2 * 4));
        __1.x = (v_.x*v_.x);
        __1.y = (v_.y*v_.y);
        __1.z = (v_.z*v_.z);
        __1.w = (v_.w*v_.w);
      *(float4*)(input_sq_local + (i_2 * 4)) = __1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tile_sum[i_3] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        tile_sum[i_3] = (tile_sum[i_3] + input_local[((rv * 8) + i_3)]);
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tile_sq_sum[i_4] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        tile_sq_sum[i_4] = (tile_sq_sum[i_4] + input_sq_local[((rv_1 * 8) + i_4)]);
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      acc_sum[i_5] = (acc_sum[i_5] + tile_sum[i_5]);
      acc_sq[i_5] = (acc_sq[i_5] + tile_sq_sum[i_5]);
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    *(float4*)(block_sum + (((((int)blockIdx.x) * 1024) + (i_6 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(acc_sum + (i_6 * 4));
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    *(float4*)(block_sq_sum + (((((int)blockIdx.x) * 1024) + (i_7 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(acc_sq + (i_7 * 4));
  }
}

