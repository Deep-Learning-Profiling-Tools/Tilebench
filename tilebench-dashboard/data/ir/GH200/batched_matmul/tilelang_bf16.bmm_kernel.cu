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

extern "C" __global__ void bmm_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) bmm_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_tile = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc[128];
  bfloat16_t C_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)A_tile)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_1 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)B_tile)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((int)blockIdx.z) * 409600) + (i_2 * 5120)) + ((((int)threadIdx.x) >> 4) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 9; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)A_tile)[((((((((k + 1) & 1) * 8192) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_3 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)B_tile)[(((((((((k + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((((int)blockIdx.z) * 409600) + (k * 40960)) + (i_4 * 5120)) + ((((int)threadIdx.x) >> 4) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 40960)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)A_tile)[((k & 1) * 8192)])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)B_tile)[((k & 1) * 8192)])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_5 * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(acc + (i_5 * 64))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)A_tile)[8192])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((bfloat16_t*)B_tile)[8192])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_1 + (((i_6 * 8192) + (ki_1 * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(acc + (i_6 * 64))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    uint1 __1;
    float2 v_ = *(float2*)(acc + (i_7 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + ((i_7 >> 5) * 40960)) + ((((int)threadIdx.x) >> 5) * 10240)) + ((i_7 & 1) * 5120)) + (((((int)threadIdx.x) & 31) >> 2) * 640)) + (((int)blockIdx.y) * 128)) + (((i_7 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}

