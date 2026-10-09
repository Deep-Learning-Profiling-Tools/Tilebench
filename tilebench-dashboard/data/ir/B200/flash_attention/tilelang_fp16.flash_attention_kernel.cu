#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 131072));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 132096));
  __shared__ __align__(16) uint64_t qk_mbar_mem[1];
  auto qk_mbar = reinterpret_cast<Barrier*>(qk_mbar_mem);
  __shared__ __align__(16) uint64_t pv_mbar_mem[1];
  auto pv_mbar = reinterpret_cast<Barrier*>(pv_mbar_mem);
  __shared__ __align__(16) uint qk_tmem[1];
  __shared__ __align__(16) uint pv_tmem[1];
  half_t q_shared_local_cast[8];
  float acc[64];
  float l_i[1];
  float m_i[1];
  float qk[64];
  float m_ij[1];
  float m_ij_clear[1];
  float alpha[1];
  float l_ij[1];
  half_t p_shared_local_cast_1[8];
  float pv[64];
  half_t O_local_cast_2[16];
  if (tl::tl_shuffle_elect<0>()) {
    qk_mbar[0].init(1);
    pv_mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(qk_tmem[0])), 128);
    tl::tmem_allocate((&(pv_tmem[0])), 128);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (((int)blockIdx.x) * 16384)) + (i * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    *(uint4*)(q_shared_local_cast + 0) = *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      uint2 __1;
      float4 __2;
        float4 __3;
        uint2 v_ = *(uint2*)(q_shared_local_cast + (vec * 4));
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 v__1 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v__1.z)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
      *(uint2*)(q_shared_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(((half_t*)q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_shared_local_cast + 0);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_2 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  l_i[0] = 0x0p+0f/*0.000000e+00*/;
  m_i[0] = -CUDART_INF_F;
  for (int k_tile = 0; k_tile < (((int)blockIdx.x) + 1); ++k_tile) {
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      *(uint4*)(((half_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (k_tile * 16384)) + (i_3 * 2048)) + (((int)threadIdx.x) * 8)));
    }
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)q_shared)[0])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)k_shared)[0])), 1, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))), uint64_t(desc_b + (((ki >> 2) * 16384) + ((ki & 3) * 32))), (*reinterpret_cast<uint32_t*>(qk_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136314896), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(qk_mbar[0])));
      }
      qk_mbar[0].wait((k_tile & 1));
    }
    tl::tcgen05_ld_32dp32bNx<64, false>(qk_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(qk[0])));
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      qk[i_4] = (((((k_tile * 128) + ((((int)threadIdx.x) >> 7) * 64)) + i_4) <= ((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127))) ? qk[i_4] : -CUDART_INF_F);
    }
    m_ij[0] = -CUDART_INF_F;
    m_ij_clear[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 64; ++rv) {
      m_ij_clear[0] = max(m_ij_clear[0], qk[rv]);
    }
    m_ij_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(m_ij_clear[0], (&(((float*)workspace)[0])));
    m_ij[0] = max(m_ij[0], m_ij_clear[0]);
    m_ij[0] = max(m_i[0], m_ij[0]);
    alpha[0] = exp2f((m_i[0] - m_ij[0]));
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      qk[i_5] = exp2f((qk[i_5] - m_ij[0]));
    }
    l_ij[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
      l_ij[0] = (l_ij[0] + qk[rv_1]);
    }
    l_ij[0] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(l_ij[0], (&(((float*)workspace_1)[0])));
    l_i[0] = ((l_i[0] * alpha[0]) + l_ij[0]);
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
        uint2 __4;
        float4 v__2 = *(float4*)(qk + ((i_6 * 8) + (vec_1 * 4)));
        ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
        *(uint2*)(p_shared_local_cast_1 + (vec_1 * 4)) = __4;
      }
      *(uint4*)(((half_t*)p_shared) + ((((((int)threadIdx.x) * 64) + ((((i_6 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_6 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_6 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(p_shared_local_cast_1 + 0);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      acc[i_7] = (acc[i_7] * alpha[0]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      *(uint4*)(((half_t*)v_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_8 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (k_tile * 16384)) + (i_8 * 2048)) + (((int)threadIdx.x) * 8)));
    }
    {
      tl::Tcgen05SMemDescriptor desc_a_1;
      tl::Tcgen05SMemDescriptor desc_b_1;
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_1, (&(((half_t*)p_shared)[0])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b_1, (&(((half_t*)v_shared)[0])), 1024, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a_1 + (((ki_1 >> 2) * 16384) + ((ki_1 & 3) * 32))), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(pv_tmem)) + 0, ((0 < ki_1) ? 1 : 0), static_cast<uint32_t>(136380432), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(pv_mbar[0])));
      }
      pv_mbar[0].wait((k_tile & 1));
    }
    tl::tcgen05_ld_32dp32bNx<64, false>(pv_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(pv[0])));
    #pragma unroll
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      acc[i_9] = (acc[i_9] + pv[i_9]);
    }
    m_i[0] = m_ij[0];
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 64; ++i_10) {
    acc[i_10] = (acc[i_10] / l_i[0]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    for (int vec_2 = 0; vec_2 < 4; ++vec_2) {
      uint2 __5;
      float4 v__3 = *(float4*)(acc + ((i_11 * 16) + (vec_2 * 4)));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(O_local_cast_2 + (vec_2 * 4)) = __5;
    }
    tl::store_global_256(&(*(ulonglong4*)(O + ((((((((int)blockIdx.z) * 83886080) + (((int)blockIdx.y) * 2621440)) + (((int)blockIdx.x) * 16384)) + ((((int)threadIdx.x) & 127) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (i_11 * 16)))), *(ulonglong4*)(O_local_cast_2 + 0));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(qk_tmem[0])), 128);
    tl::tmem_deallocate((&(pv_tmem[0])), 128);
  }
}

