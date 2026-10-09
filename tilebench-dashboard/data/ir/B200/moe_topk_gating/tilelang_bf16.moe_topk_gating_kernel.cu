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

extern "C" __global__ void moe_topk_gating_kernel_kernel(const bfloat16_t* __restrict__ logits, int* __restrict__ topk_idx, bfloat16_t* __restrict__ topk_w);
extern "C" __global__ void __launch_bounds__(32, 1) moe_topk_gating_kernel_kernel(const bfloat16_t* __restrict__ logits, int* __restrict__ topk_idx, bfloat16_t* __restrict__ topk_w) {
  float logits_reg[4];
  float topk_vals[1];
  int topk_idxs[1];
  bfloat16_t logits_local_cast[4];
  float curr_max_val[1];
  int src_idx[4];
  int curr_max_idx[1];
  float mx[1];
  float rs[1];
  float broadcast_var = -CUDART_INF_F;
  *(float4*)(logits_reg + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  topk_vals[0] = -CUDART_INF_F;
  topk_idxs[0] = 0;
  *(uint2*)(logits_local_cast + 0) = *(uint2*)(logits + ((((int)blockIdx.x) * 128) + (((int)threadIdx.x) * 4)));
  float4 __1;
  uint2 v_ = *(uint2*)(logits_local_cast + 0);
  ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
  ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
  *(float4*)(logits_reg + 0) = __1;
  for (int i = 0; i < 2; ++i) {
    curr_max_val[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      curr_max_val[0] = max(curr_max_val[0], logits_reg[rv]);
    }
    curr_max_val[0] = tl::AllReduce<tl::MaxOp, 32, 1, 0, tl::NamedBarrier<32>>::run(curr_max_val[0]);
    int broadcast_var_1 = 128;
    *(int4*)(src_idx + 0) = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      src_idx[i_1] = ((logits_reg[i_1] == curr_max_val[0]) ? ((((int)threadIdx.x) * 4) + i_1) : 128);
    }
    curr_max_idx[0] = 2147483647;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      curr_max_idx[0] = min(curr_max_idx[0], src_idx[rv_1]);
    }
    curr_max_idx[0] = tl::AllReduce<tl::MinOp, 32, 1, 0, tl::NamedBarrier<32>>::run(curr_max_idx[0]);
    topk_vals[0] = (((((int)threadIdx.x) & 1) == i) ? curr_max_val[0] : topk_vals[0]);
    topk_idxs[0] = (((((int)threadIdx.x) & 1) == i) ? curr_max_idx[0] : topk_idxs[0]);
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      logits_reg[i_2] = ((((((int)threadIdx.x) * 4) + i_2) == curr_max_idx[0]) ? -CUDART_INF_F : logits_reg[i_2]);
    }
  }
  mx[0] = -CUDART_INF_F;
  mx[0] = max(mx[0], topk_vals[0]);
  mx[0] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<32>>::run(mx[0]);
  topk_vals[0] = expf((topk_vals[0] - mx[0]));
  rs[0] = 0x0p+0f/*0.000000e+00*/;
  rs[0] = (rs[0] + topk_vals[0]);
  rs[0] = tl::AllReduce<tl::SumOp, 2, 1, 0, tl::NamedBarrier<32>>::run(rs[0]);
  topk_vals[0] = (topk_vals[0] / rs[0]);
  topk_w[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))] = ((bfloat16_t)topk_vals[0]);
  if ((((int)threadIdx.x) >> 1) == 0) {
    topk_idx[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))] = topk_idxs[0];
  }
}

