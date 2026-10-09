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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  half_t q_shared_local_cast[8];
  float acc[64];
  float l_i[2];
  float m_i[2];
  float qk[64];
  float m_ij[2];
  float m_ij_clear[2];
  float alpha[2];
  float l_ij[2];
  float pv[64];
  half_t O_local_cast_1[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (((int)blockIdx.x) * 8192)) + (i * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    __syncthreads();
    *(uint4*)(q_shared_local_cast + 0) = *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      uint2 __1;
      float4 __2;
        float4 __3;
        uint2 v_ = *(uint2*)(q_shared_local_cast + (vec * 4));
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 v__1 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        __2.x = (__3.x*v__1.x);
        __2.y = (__3.y*v__1.y);
        __2.z = (__3.z*v__1.z);
        __2.w = (__3.w*v__1.w);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
      *(uint2*)(q_shared_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_shared_local_cast + 0);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_2 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(l_i + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(m_i + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  for (int k_tile = 0; k_tile < ((((int)blockIdx.x) >> 1) + 1); ++k_tile) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      *(uint4*)(((half_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (k_tile * 16384)) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)));
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)k_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(qk + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qk + 0), 64);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      qk[i_4] = ((((((k_tile * 128) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) ? qk[i_4] : -CUDART_INF_F);
    }
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(m_ij + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      m_ij_clear[i_5] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        m_ij_clear[i_5] = max(m_ij_clear[i_5], qk[((((rv & 15) * 4) + (i_5 * 2)) + (rv >> 4))]);
      }
      m_ij_clear[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_ij_clear[i_5]);
      m_ij[i_5] = max(m_ij[i_5], m_ij_clear[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      m_ij[i_6] = max(m_i[i_6], m_ij[i_6]);
      alpha[i_6] = exp2f((m_i[i_6] - m_ij[i_6]));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      qk[i_7] = exp2f((qk[i_7] - m_ij[((i_7 & 3) >> 1)]));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      l_ij[i_8] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        l_ij[i_8] = (l_ij[i_8] + qk[((((rv_1 & 15) * 4) + (i_8 * 2)) + (rv_1 >> 4))]);
      }
      l_ij[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(l_ij[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      l_i[i_9] = ((l_i[i_9] * alpha[i_9]) + l_ij[i_9]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      tl::ptx_stmatrix_x4((&(((half_t*)p_shared)[(((((i_10 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)qk[(i_10 * 8)]), ((half_t)qk[((i_10 * 8) + 1)])), __pack_half2(((half_t)qk[((i_10 * 8) + 2)]), ((half_t)qk[((i_10 * 8) + 3)])), __pack_half2(((half_t)qk[((i_10 * 8) + 4)]), ((half_t)qk[((i_10 * 8) + 5)])), __pack_half2(((half_t)qk[((i_10 * 8) + 6)]), ((half_t)qk[((i_10 * 8) + 7)])));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 64; ++i_11) {
      acc[i_11] = (acc[i_11] * alpha[((i_11 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      *(uint4*)(((half_t*)v_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_12 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (k_tile * 16384)) + (i_12 * 1024)) + (((int)threadIdx.x) * 8)));
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)p_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_1, (&(((half_t*)v_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(pv + 0)), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(pv + 0), 64);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 64; ++i_13) {
      acc[i_13] = (acc[i_13] + pv[i_13]);
    }
    *(float2*)(m_i + 0) = *(float2*)(m_ij + 0);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 64; ++i_14) {
    acc[i_14] = (acc[i_14] / l_i[((i_14 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 32; ++i_15) {
    uint1 __4;
    float2 v__2 = *(float2*)(acc + (i_15 * 2));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
    *(uint1*)(O_local_cast_1 + 0) = __4;
    *(uint1*)(O + ((((((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (((int)blockIdx.x) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_15 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_local_cast_1 + 0);
  }
}

