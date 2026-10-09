#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const half_t* __restrict__ Q, const half_t* __restrict__ V, const int* __restrict__ col_indices, const int* __restrict__ row_indices);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap O_desc, const half_t* __restrict__ Q, const half_t* __restrict__ V, const int* __restrict__ col_indices, const int* __restrict__ row_indices) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* o_shared2 = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_shared2 = ((void*)((char*)buf_dyn_shmem + 8192));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* k_shared2 = ((void*)((char*)buf_dyn_shmem + 24576));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* v_shared2 = ((void*)((char*)buf_dyn_shmem + 49152));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 57344));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 58368));
  float acc[16];
  float acc2[16];
  float logsum[2];
  float scores_max[2];
  int start_l = 0;
  int end_l = 0;
  int l = 0;
  int col_idx = 0;
  int start_n = 0;
  float qk[16];
  float scores_max_prev[2];
  float alpha[2];
  float scores_sum[2];
  float pv[16];
  float pv2[16];
  float scores_max_clear[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(O_desc);
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(((half_t*)q_shared) + (((((i * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.y) * 1310720) + (((int)blockIdx.x) * 8192)) + (i * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((half_t*)q_shared2) + (((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((((int)blockIdx.y) * 1310720) + (((int)blockIdx.x) * 8192)) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc2 + (i_3 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  start_l = row_indices[((int)blockIdx.x)];
  end_l = row_indices[(((int)blockIdx.x) + 1)];
  l = start_l;
  while (1) {
    if (!((l < end_l))) { break; }
    int condval;
    if (((0 <= l) && (l < 25600))) {
      condval = col_indices[l];
    } else {
      condval = 0;
    }
    col_idx = condval;
    start_n = (col_idx * 64);
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_4) < 320) && (0 <= (((i_4 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_4 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_4) < 320))) {
        condval_1 = *(uint4*)(K + ((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_4) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((half_t*)k_shared) + (((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)k_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 4096) + (ki * 32)) >> 4)), ((uint32_t*)(qk + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_5) < 320) && (0 <= (((i_5 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_5 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_5) < 320))) {
        condval_2 = *(uint4*)(K + (((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_5) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_2 = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(((half_t*)k_shared2) + (((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)q_shared2)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)k_shared2)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_1 * 32)) >> 4)), ((uint32_t*)(qk + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      qk[i_6] = (((((((((((int)threadIdx.x) >> 7) * 32) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + start_n) + (i_6 & 1)) < 10240) && (((((((((int)threadIdx.x) >> 7) * 32) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + start_n) + (i_6 & 1)) <= ((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) ? qk[i_6] : -CUDART_INF_F);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      qk[i_7] = (qk[i_7] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_max_clear[i_8] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_8] = max(scores_max_clear[i_8], qk[((((rv & 3) * 4) + (i_8 * 2)) + (rv >> 2))]);
      }
      scores_max_clear[i_8] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_8], (&(((float*)workspace_1)[0])));
      scores_max_clear[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_8]);
      scores_max[i_8] = max(scores_max[i_8], scores_max_clear[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      bool has_prev = (0x0p+0f/*0.000000e+00*/ < logsum[i_9]);
      bool has_valid = ((start_n < 10240) && (start_n <= ((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (i_9 * 8)) + ((((int)threadIdx.x) & 31) >> 2))));
      scores_max[i_9] = max(scores_max[i_9], scores_max_prev[i_9]);
      float m_safe = ((has_prev || has_valid) ? scores_max[i_9] : 0x0p+0f/*0.000000e+00*/);
      alpha[i_9] = (has_prev ? expf((scores_max_prev[i_9] - m_safe)) : 0x0p+0f/*0.000000e+00*/);
      scores_max[i_9] = ((has_prev || has_valid) ? scores_max[i_9] : scores_max_prev[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      float m_safe_1 = (((0x0p+0f/*0.000000e+00*/ < logsum[((i_10 & 3) >> 1)]) || ((start_n < 10240) && (start_n <= ((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))))) ? scores_max[((i_10 & 3) >> 1)] : 0x0p+0f/*0.000000e+00*/);
      qk[i_10] = (((((((((((int)threadIdx.x) >> 7) * 32) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + start_n) + (i_10 & 1)) < 10240) && (((((((((int)threadIdx.x) >> 7) * 32) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + start_n) + (i_10 & 1)) <= ((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) ? expf((qk[i_10] - m_safe_1)) : 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_sum[i_11] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_11] = (scores_sum[i_11] + qk[((((rv_1 & 3) * 4) + (i_11 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_11] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11], (&(((float*)workspace)[0])));
      scores_sum[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      logsum[i_12] = ((logsum[i_12] * alpha[i_12]) + scores_sum[i_12]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      tl::ptx_stmatrix_x4((&(((half_t*)p_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)qk[(i_13 * 8)]), ((half_t)qk[((i_13 * 8) + 1)])), __pack_half2(((half_t)qk[((i_13 * 8) + 2)]), ((half_t)qk[((i_13 * 8) + 3)])), __pack_half2(((half_t)qk[((i_13 * 8) + 4)]), ((half_t)qk[((i_13 * 8) + 5)])), __pack_half2(((half_t)qk[((i_13 * 8) + 6)]), ((half_t)qk[((i_13 * 8) + 7)])));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      acc[i_14] = (acc[i_14] * alpha[((i_14 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      acc2[i_15] = (acc2[i_15] * alpha[((i_15 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_16) < 320) && (0 <= (((i_16 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_16 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_16) < 320))) {
        condval_3 = *(uint4*)(V + ((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_16) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_3 = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
      }
      *(uint4*)(((half_t*)v_shared) + (((((((((int)threadIdx.x) & 7) >> 2) * 2048) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)p_shared)[0])));
      tl::initialize_wgmma_descriptor<2, 512, 32>(desc_b_2, (&(((half_t*)v_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_2 * 1024)) >> 4)), ((uint32_t*)(pv + 0)), ((0 < ki_2) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      acc[i_17] = (acc[i_17] + pv[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      half_t broadcast_var_8 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_18) < 320) && (0 <= (((i_18 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_18 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_18) < 320))) {
        condval_4 = *(uint4*)(V + (((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_18) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_4 = make_uint4(__pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8));
      }
      *(uint4*)(((half_t*)v_shared2) + (((((((((int)threadIdx.x) & 7) >> 2) * 2048) + (i_18 * 1024)) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
    }
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)p_shared)[0])));
      tl::initialize_wgmma_descriptor<2, 512, 32>(desc_b_3, (&(((half_t*)v_shared2)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv2 + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_3 * 1024)) >> 4)), ((uint32_t*)(pv2 + 0)), ((0 < ki_3) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv2 + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      acc2[i_19] = (acc2[i_19] + pv2[i_19]);
    }
    l = (l + 1);
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 16; ++i_20) {
    acc[i_20] = (acc[i_20] / ((0x0p+0f/*0.000000e+00*/ < logsum[((i_20 & 3) >> 1)]) ? logsum[((i_20 & 3) >> 1)] : 0x1p+0f/*1.000000e+00*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    tl::ptx_stmatrix_x4((&(((half_t*)o_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc[(i_21 * 8)]), ((half_t)acc[((i_21 * 8) + 1)])), __pack_half2(((half_t)acc[((i_21 * 8) + 2)]), ((half_t)acc[((i_21 * 8) + 3)])), __pack_half2(((half_t)acc[((i_21 * 8) + 4)]), ((half_t)acc[((i_21 * 8) + 5)])), __pack_half2(((half_t)acc[((i_21 * 8) + 6)]), ((half_t)acc[((i_21 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((half_t*)o_shared)[0])), 0, (((int)blockIdx.x) * 64), (((int)blockIdx.y) & 7), (((int)blockIdx.y) >> 3));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 16; ++i_22) {
    acc2[i_22] = (acc2[i_22] / ((0x0p+0f/*0.000000e+00*/ < logsum[((i_22 & 3) >> 1)]) ? logsum[((i_22 & 3) >> 1)] : 0x1p+0f/*1.000000e+00*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    tl::ptx_stmatrix_x4((&(((half_t*)o_shared2)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_23) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc2[(i_23 * 8)]), ((half_t)acc2[((i_23 * 8) + 1)])), __pack_half2(((half_t)acc2[((i_23 * 8) + 2)]), ((half_t)acc2[((i_23 * 8) + 3)])), __pack_half2(((half_t)acc2[((i_23 * 8) + 4)]), ((half_t)acc2[((i_23 * 8) + 5)])), __pack_half2(((half_t)acc2[((i_23 * 8) + 6)]), ((half_t)acc2[((i_23 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((half_t*)o_shared2)[0])), 64, (((int)blockIdx.x) * 64), (((int)blockIdx.y) & 7), (((int)blockIdx.y) >> 3));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

