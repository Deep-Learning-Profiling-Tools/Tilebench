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

extern "C" __global__ void flash_decode_stage2_kernel_kernel(const int* __restrict__ b_seqlen, const float* __restrict__ mid_o, const float* __restrict__ mid_o_lse, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(64, 1) flash_decode_stage2_kernel_kernel(const int* __restrict__ b_seqlen, const float* __restrict__ mid_o, const float* __restrict__ mid_o_lse, float* __restrict__ output) {
  float sum_exp = 0x0p+0f/*0.000000e+00*/;
  float max_logic = 0x0p+0f/*0.000000e+00*/;
  float acc[2];
  float out[2];
  float tv[2];
  float exp_logic = 0x0p+0f/*0.000000e+00*/;
  sum_exp = 0x0p+0f/*0.000000e+00*/;
  max_logic = -CUDART_INF_F;
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(acc + 0) = make_float2(broadcast_var, broadcast_var);
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(out + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  int cur_batch_seq_len = b_seqlen[((int)blockIdx.x)];
  for (int block_seq_n = 0; block_seq_n < ((cur_batch_seq_len <= 0) ? 0 : ((cur_batch_seq_len + 127) >> 7)); ++block_seq_n) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    float2 condval;
    if ((block_seq_n < 320)) {
      condval = *(float2*)(mid_o + ((((((int64_t)((int)blockIdx.x)) * (int64_t)327680) + (((int64_t)((int)blockIdx.y)) * (int64_t)40960)) + (((int64_t)block_seq_n) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
    } else {
      condval = make_float2(broadcast_var_2, broadcast_var_2);
    }
    *(float2*)(tv + 0) = condval;
    float condval_1;
    if ((block_seq_n < 320)) {
      condval_1 = mid_o_lse[(((((int)blockIdx.x) * 2560) + (((int)blockIdx.y) * 320)) + block_seq_n)];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float tlogic = condval_1;
    float new_max_logic = max(tlogic, max_logic);
    float old_scale = expf((max_logic - new_max_logic));
    exp_logic = expf((tlogic - new_max_logic));
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc[i] = ((acc[i] * old_scale) + (exp_logic * tv[i]));
    }
    sum_exp = ((sum_exp * old_scale) + exp_logic);
    max_logic = new_max_logic;
  }
  if (0 < cur_batch_seq_len) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      out[i_1] = (acc[i_1] / sum_exp);
    }
  }
  *(float2*)(output + (((((int)blockIdx.x) * 1024) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.x) * 2))) = *(float2*)(out + 0);
}

