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
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 59392));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 60416));
  __shared__ __align__(16) uint64_t qk_mbar_mem[1];
  auto qk_mbar = reinterpret_cast<Barrier*>(qk_mbar_mem);
  __shared__ __align__(16) uint64_t pv_mbar_mem[1];
  auto pv_mbar = reinterpret_cast<Barrier*>(pv_mbar_mem);
  __shared__ __align__(16) uint qk_tmem[1];
  __shared__ __align__(16) uint pv2_tmem[1];
  __shared__ __align__(16) uint pv_tmem[1];
  float acc[16];
  float acc2[16];
  float logsum[1];
  float scores_max[1];
  int start_l = 0;
  int end_l = 0;
  int l = 0;
  int col_idx = 0;
  int start_n = 0;
  float qk[16];
  float scores_max_prev[1];
  float alpha[1];
  float scores_sum[1];
  float pv[16];
  float pv2[16];
  float scores_max_clear[1];
  half_t p_shared_local_cast[8];
  half_t o_shared_local_cast_1[8];
  half_t o_shared2_local_cast_2[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(O_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    qk_mbar[0].init(1);
    pv_mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(qk_tmem[0])), 32);
    tl::tmem_allocate((&(pv2_tmem[0])), 32);
    tl::tmem_allocate((&(pv_tmem[0])), 32);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
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
  logsum[0] = 0x0p+0f/*0.000000e+00*/;
  scores_max[0] = -CUDART_INF_F;
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
      half_t broadcast_var_2 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_4) < 320) && (0 <= (((i_4 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_4 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_4) < 320))) {
        condval_1 = *(uint4*)(K + ((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_4) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((half_t*)k_shared) + (((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
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
        for (int ki = 0; ki < 4; ++ki) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat16>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)), (*reinterpret_cast<uint32_t*>(qk_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(68157456), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(qk_mbar[0])));
      }
      qk_mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_5) < 320) && (0 <= (((i_5 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_5 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_5) < 320))) {
        condval_2 = *(uint4*)(K + (((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_5) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_2 = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((half_t*)k_shared2) + (((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    {
      tl::Tcgen05SMemDescriptor desc_a_1;
      tl::Tcgen05SMemDescriptor desc_b_1;
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_1, (&(((half_t*)q_shared2)[0])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b_1, (&(((half_t*)k_shared2)[0])), 1, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat16>(uint64_t(desc_a_1 + (ki_1 * 32)), uint64_t(desc_b_1 + (ki_1 * 32)), (*reinterpret_cast<uint32_t*>(qk_tmem)) + 0, 1, static_cast<uint32_t>(68157456), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(qk_mbar[0])));
      }
      qk_mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_ld_32dp32bNx<16, false>(qk_tmem[0], ((((int)threadIdx.x) >> 7) * 16), (&(qk[0])));
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      qk[i_6] = (((((((((((int)threadIdx.x) & 127) >> 6) * 32) + ((((int)threadIdx.x) >> 7) * 16)) + start_n) + i_6) < 10240) && (((((((((int)threadIdx.x) & 127) >> 6) * 32) + ((((int)threadIdx.x) >> 7) * 16)) + start_n) + i_6) <= ((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)))) ? qk[i_6] : -CUDART_INF_F);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      qk[i_7] = (qk[i_7] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
    }
    scores_max_prev[0] = scores_max[0];
    scores_max[0] = -CUDART_INF_F;
    scores_max_clear[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      scores_max_clear[0] = max(scores_max_clear[0], qk[rv]);
    }
    scores_max_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[0], (&(((float*)workspace_1)[0])));
    scores_max_clear[0] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<256>>::run(scores_max_clear[0], (&(((float*)workspace_2)[0])));
    scores_max[0] = max(scores_max[0], scores_max_clear[0]);
    bool has_prev = (0x0p+0f/*0.000000e+00*/ < logsum[0]);
    bool has_valid = ((start_n < 10240) && (start_n <= ((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63))));
    scores_max[0] = max(scores_max[0], scores_max_prev[0]);
    float m_safe = ((has_prev || has_valid) ? scores_max[0] : 0x0p+0f/*0.000000e+00*/);
    alpha[0] = (has_prev ? expf((scores_max_prev[0] - m_safe)) : 0x0p+0f/*0.000000e+00*/);
    scores_max[0] = ((has_prev || has_valid) ? scores_max[0] : scores_max_prev[0]);
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float m_safe_1 = (((0x0p+0f/*0.000000e+00*/ < logsum[0]) || ((start_n < 10240) && (start_n <= ((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63))))) ? scores_max[0] : 0x0p+0f/*0.000000e+00*/);
      qk[i_8] = (((((((((((int)threadIdx.x) & 127) >> 6) * 32) + ((((int)threadIdx.x) >> 7) * 16)) + start_n) + i_8) < 10240) && (((((((((int)threadIdx.x) & 127) >> 6) * 32) + ((((int)threadIdx.x) >> 7) * 16)) + start_n) + i_8) <= ((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)))) ? expf((qk[i_8] - m_safe_1)) : 0x0p+0f/*0.000000e+00*/);
    }
    scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      scores_sum[0] = (scores_sum[0] + qk[rv_1]);
    }
    scores_sum[0] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[0], (&(((float*)workspace_3)[0])));
    scores_sum[0] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<256>>::run(scores_sum[0], (&(((float*)workspace)[0])));
    logsum[0] = ((logsum[0] * alpha[0]) + scores_sum[0]);
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(qk + ((i_9 * 8) + (vec * 4)));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(p_shared_local_cast + (vec * 4)) = __1;
      }
      *(uint4*)(((half_t*)p_shared) + (((((((int)threadIdx.x) & 63) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((i_9 + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(p_shared_local_cast + 0);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      acc[i_10] = (acc[i_10] * alpha[0]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      acc2[i_11] = (acc2[i_11] * alpha[0]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_12) < 320) && (0 <= (((i_12 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_12 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_12) < 320))) {
        condval_3 = *(uint4*)(V + ((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_12) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_3 = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((half_t*)v_shared) + (((((i_12 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
    }
    {
      tl::Tcgen05SMemDescriptor desc_a_2;
      tl::Tcgen05SMemDescriptor desc_b_2;
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_2, (&(((half_t*)p_shared)[0])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b_2, (&(((half_t*)v_shared)[0])), 0, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat16>(uint64_t(desc_a_2 + (ki_2 * 32)), uint64_t(desc_b_2 + (ki_2 * 2048)), (*reinterpret_cast<uint32_t*>(pv_tmem)) + 0, ((0 < ki_2) ? 1 : 0), static_cast<uint32_t>(68222992), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(pv_mbar[0])));
      }
      pv_mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_ld_32dp32bNx<16, false>(pv_tmem[0], ((((int)threadIdx.x) >> 7) * 16), (&(pv[0])));
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      acc[i_13] = (acc[i_13] + pv[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_14) < 320) && (0 <= (((i_14 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (0 <= (((i_14 * 32) + (((int)threadIdx.x) >> 3)) + start_n))) && (((((((int)threadIdx.x) >> 3) + start_n) >> 5) + i_14) < 320))) {
        condval_4 = *(uint4*)(V + (((((((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)1310720) + (((int64_t)i_14) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)start_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_4 = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(((half_t*)v_shared2) + (((((i_14 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
    }
    {
      tl::Tcgen05SMemDescriptor desc_a_3;
      tl::Tcgen05SMemDescriptor desc_b_3;
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_3, (&(((half_t*)p_shared)[0])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b_3, (&(((half_t*)v_shared2)[0])), 0, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat16>(uint64_t(desc_a_3 + (ki_3 * 32)), uint64_t(desc_b_3 + (ki_3 * 2048)), (*reinterpret_cast<uint32_t*>(pv2_tmem)) + 0, ((0 < ki_3) ? 1 : 0), static_cast<uint32_t>(68222992), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(pv_mbar[0])));
      }
      pv_mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_ld_32dp32bNx<16, false>(pv2_tmem[0], ((((int)threadIdx.x) >> 7) * 16), (&(pv2[0])));
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      acc2[i_15] = (acc2[i_15] + pv2[i_15]);
    }
    l = (l + 1);
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    acc[i_16] = (acc[i_16] / ((0x0p+0f/*0.000000e+00*/ < logsum[0]) ? logsum[0] : 0x1p+0f/*1.000000e+00*/));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc + ((i_17 * 8) + (vec_1 * 4)));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(o_shared_local_cast_1 + (vec_1 * 4)) = __2;
    }
    *(uint4*)(((half_t*)o_shared) + (((((((int)threadIdx.x) & 63) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((i_17 + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(o_shared_local_cast_1 + 0);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((half_t*)o_shared)[0])), 0, (((int)blockIdx.x) * 64), (((int)blockIdx.y) & 7), (((int)blockIdx.y) >> 3));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 16; ++i_18) {
    acc2[i_18] = (acc2[i_18] / ((0x0p+0f/*0.000000e+00*/ < logsum[0]) ? logsum[0] : 0x1p+0f/*1.000000e+00*/));
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc2 + ((i_19 * 8) + (vec_2 * 4)));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
      *(uint2*)(o_shared2_local_cast_2 + (vec_2 * 4)) = __3;
    }
    *(uint4*)(((half_t*)o_shared2) + (((((((int)threadIdx.x) & 63) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((i_19 + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(o_shared2_local_cast_2 + 0);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store(O_desc, (&(((half_t*)o_shared2)[0])), 64, (((int)blockIdx.x) * 64), (((int)blockIdx.y) & 7), (((int)blockIdx.y) >> 3));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(qk_tmem[0])), 32);
    tl::tmem_deallocate((&(pv2_tmem[0])), 32);
    tl::tmem_deallocate((&(pv_tmem[0])), 32);
  }
}

