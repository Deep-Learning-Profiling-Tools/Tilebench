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

extern "C" __global__ void matmul_kernel_kernel(const half_t* __restrict__ a, const half_t* __restrict__ b, half_t* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel_kernel(const half_t* __restrict__ a, const half_t* __restrict__ b, half_t* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 49152));
  float acc[128];
  half_t c_local_cast[2];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)a_tile)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(a[((((((int)blockIdx.x) * 2621440) + (i_1 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)b_tile)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(b[((((i_2 * 32768) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)a_tile)[((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(a[(((((((int)blockIdx.x) * 2621440) + (i_3 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)b_tile)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(b[(((((i_4 * 32768) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 318; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)a_tile)[((((((((k + 2) % 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(a[((((((((int)blockIdx.x) * 2621440) + (i_5 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)b_tile)[(((((((((k + 2) % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(b[((((((k * 262144) + (i_6 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)a_tile)[((k % 3) * 8192)])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((half_t*)b_tile)[((k % 3) * 8192)])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_7 * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(acc + (i_7 * 64))), 1);
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
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)a_tile)[0])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)b_tile)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_1 + (((i_8 * 8192) + (ki_1 * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(acc + (i_8 * 64))), 1);
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
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)a_tile)[8192])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)b_tile)[8192])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_2 + (((i_9 * 8192) + (ki_2 * 32)) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(acc + (i_9 * 64))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 64; ++i_10) {
    uint1 __1;
    float2 v_ = *(float2*)(acc + (i_10 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(c_local_cast + 0) = __1;
    *(uint1*)(c + ((((((((((int)blockIdx.x) * 524288) + ((i_10 >> 5) * 262144)) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_10 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (((i_10 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(c_local_cast + 0);
  }
}

