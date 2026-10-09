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

extern "C" __global__ void full_tiles_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) full_tiles_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float acc[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.x) + 212) / 1792) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((i_2 * 229376) + ((((int)threadIdx.x) >> 4) * 28672)) + ((((((int)blockIdx.x) + 212) % 1792) >> 3) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[((((((((((int)blockIdx.x) + 212) / 1792) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_3 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B[(((((i_4 * 229376) + ((((int)threadIdx.x) >> 4) * 28672)) + ((((((int)blockIdx.x) + 212) % 1792) >> 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 917504)])));
  }
  tl::cp_async_commit();
  for (int k_tile = 0; k_tile < 126; ++k_tile) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[(((((((k_tile + 2) % 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((((int)blockIdx.x) + 212) / 1792) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_5 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_tile * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[(((((((((k_tile + 2) % 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((k_tile * 917504) + (i_6 * 229376)) + ((((int)threadIdx.x) >> 4) * 28672)) + ((((((int)blockIdx.x) + 212) % 1792) >> 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 1835008)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((bfloat16_t*)a_shared)[((k_tile % 3) * 4096)])));
      tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((bfloat16_t*)b_shared)[((k_tile % 3) * 4096)])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_7 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(acc + (i_7 * 64))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a_1, (&(((bfloat16_t*)a_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b_1, (&(((bfloat16_t*)b_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_1 + (((i_8 * 4096) + (ki_1 * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(acc + (i_8 * 64))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a_2, (&(((bfloat16_t*)a_shared)[4096])));
    tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b_2, (&(((bfloat16_t*)b_shared)[4096])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_2 + (((i_9 * 4096) + (ki_2 * 32)) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(acc + (i_9 * 64))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 64; ++i_10) {
    *(float2*)(C + (((((((((((((int)blockIdx.x) + 212) / 1792) * 29360128) + (((((int)blockIdx.x) + 4) & 7) * 3670016)) + ((i_10 >> 5) * 1835008)) + ((((int)threadIdx.x) >> 5) * 458752)) + ((i_10 & 1) * 229376)) + (((((int)threadIdx.x) & 31) >> 2) * 28672)) + ((((((int)blockIdx.x) + 212) % 1792) >> 3) * 128)) + (((i_10 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_10 * 2));
  }
}

