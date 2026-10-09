#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
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

extern "C" __global__ void out_kernel_kernel(float* __restrict__ O, const float* __restrict__ PhiQ, const float* __restrict__ S, const float* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(128, 1) out_kernel_kernel(float* __restrict__ O, const float* __restrict__ PhiQ, const float* __restrict__ S, const float* __restrict__ Z) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 81920));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 81920));
  float numer[64];
  float denom[2];
  float q_frag[64];
  float z_frag[1];
  float denom_prod[64];
  float denom_delta[2];
  float denom_delta_clear[64];
  float denom_delta_clear_1[64];
  float denom_delta_clear_2[64];
  float out[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(numer + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denom + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    ((tfloat32_t*)s_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_1 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_1 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)S[(((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.y) * 128)) + (i_1 * 2)) + (((int)threadIdx.x) >> 6))]);
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    ((tfloat32_t*)s_shared)[((((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_2 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3)) + 8192)] = ((tfloat32_t)S[((((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.y) * 128)) + (i_2 * 2)) + (((int)threadIdx.x) >> 6)) + 16384)]);
  }
  tl::cp_async_commit();
  for (int k_tile = 0; k_tile < 2; ++k_tile) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      float condval;
      if ((((((int)blockIdx.x) * 4) + (i_3 >> 4)) < 625)) {
        condval = PhiQ[((((((int)blockIdx.x) * 16384) + (i_3 * 256)) + (k_tile * 64)) + (((int)threadIdx.x) & 63))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      q_frag[i_3] = condval;
    }
    __syncthreads();
    if ((((int)threadIdx.x) >> 6) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 64; ++i_4) {
        ((tfloat32_t*)q_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_4 * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_4]);
      }
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((tfloat32_t*)q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((tfloat32_t*)s_shared)[(k_tile * 8192)])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 64, 128, 8, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(numer + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      ((tfloat32_t*)s_shared)[((((((((k_tile * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_5 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_5 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)S[((((((k_tile * 16384) + ((((int)threadIdx.x) & 63) * 256)) + (((int)blockIdx.y) * 128)) + (i_5 * 2)) + (((int)threadIdx.x) >> 6)) + 32768)]);
    }
    tl::cp_async_commit();
    z_frag[0] = Z[((k_tile * 64) + (((int)threadIdx.x) & 63))];
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      denom_prod[i_6] = (q_frag[i_6] * z_frag[0]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      denom_delta_clear[i_7] = 0x0p+0f/*0.000000e+00*/;
      denom_delta_clear[i_7] = (denom_delta_clear[i_7] + denom_prod[i_7]);
      denom_delta_clear[i_7] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear[i_7], (&(((float*)workspace_2)[0])));
      if (((((((int)threadIdx.x) >> 5) * 16) + (((i_7 & 15) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_7) {
        denom_delta[((i_7 & 15) >> 3)] = denom_delta_clear[i_7];
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      denom[i_8] = (denom[i_8] + denom_delta[i_8]);
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 64; ++i_9) {
    float condval_1;
    if ((((((int)blockIdx.x) * 4) + (i_9 >> 4)) < 625)) {
      condval_1 = PhiQ[((((((int)blockIdx.x) * 16384) + (i_9 * 256)) + (((int)threadIdx.x) & 63)) + 128)];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    q_frag[i_9] = condval_1;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 6) == 0) {
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      ((tfloat32_t*)q_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_10]);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((tfloat32_t*)q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((tfloat32_t*)s_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::wgmma_ss<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 64, 128, 8, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 16384) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(numer + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
  }
  z_frag[0] = Z[((((int)threadIdx.x) & 63) + 128)];
  #pragma unroll
  for (int i_11 = 0; i_11 < 64; ++i_11) {
    denom_prod[i_11] = (q_frag[i_11] * z_frag[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 64; ++i_12) {
    denom_delta_clear_1[i_12] = 0x0p+0f/*0.000000e+00*/;
    denom_delta_clear_1[i_12] = (denom_delta_clear_1[i_12] + denom_prod[i_12]);
    denom_delta_clear_1[i_12] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear_1[i_12], (&(((float*)workspace_1)[0])));
    if (((((((int)threadIdx.x) >> 5) * 16) + (((i_12 & 15) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_12) {
      denom_delta[((i_12 & 15) >> 3)] = denom_delta_clear_1[i_12];
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    denom[i_13] = (denom[i_13] + denom_delta[i_13]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 64; ++i_14) {
    float condval_2;
    if ((((((int)blockIdx.x) * 4) + (i_14 >> 4)) < 625)) {
      condval_2 = PhiQ[((((((int)blockIdx.x) * 16384) + (i_14 * 256)) + (((int)threadIdx.x) & 63)) + 192)];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    q_frag[i_14] = condval_2;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 6) == 0) {
    #pragma unroll
    for (int i_15 = 0; i_15 < 64; ++i_15) {
      ((tfloat32_t*)q_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_15 * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_15 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_15 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_15]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((tfloat32_t*)q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((tfloat32_t*)s_shared)[8192])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 64, 128, 8, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(numer + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(numer + 0), 64);
  }
  z_frag[0] = Z[((((int)threadIdx.x) & 63) + 192)];
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    denom_prod[i_16] = (q_frag[i_16] * z_frag[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 64; ++i_17) {
    denom_delta_clear_2[i_17] = 0x0p+0f/*0.000000e+00*/;
    denom_delta_clear_2[i_17] = (denom_delta_clear_2[i_17] + denom_prod[i_17]);
    denom_delta_clear_2[i_17] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear_2[i_17], (&(((float*)workspace)[0])));
    if (((((((int)threadIdx.x) >> 5) * 16) + (((i_17 & 15) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_17) {
      denom_delta[((i_17 & 15) >> 3)] = denom_delta_clear_2[i_17];
    }
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    denom[i_18] = (denom[i_18] + denom_delta[i_18]);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 64; ++i_19) {
    out[i_19] = (numer[i_19] / (denom[((i_19 & 3) >> 1)] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 5)) < 625) {
    #pragma unroll
    for (int i_20 = 0; i_20 < 32; ++i_20) {
      *(float2*)(O + (((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + ((i_20 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.y) * 128)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(out + (i_20 * 2));
    }
  }
}

