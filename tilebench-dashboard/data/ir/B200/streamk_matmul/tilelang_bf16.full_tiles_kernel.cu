#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
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
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  float acc[256];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 256);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.x) + 212) / 896) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((i_1 * 114688) + ((((int)threadIdx.x) >> 5) * 28672)) + ((((((int)blockIdx.x) + 212) % 896) >> 3) * 256)) + ((((int)threadIdx.x) & 31) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[((((((((((int)blockIdx.x) + 212) / 896) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_3 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[(((((i_3 * 114688) + ((((int)threadIdx.x) >> 5) * 28672)) + ((((((int)blockIdx.x) + 212) % 896) >> 3) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 917504)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[(((((i_4 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[((((((((((int)blockIdx.x) + 212) / 896) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_4 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_5 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B[(((((i_5 * 114688) + ((((int)threadIdx.x) >> 5) * 28672)) + ((((((int)blockIdx.x) + 212) % 896) >> 3) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 1835008)])));
  }
  tl::cp_async_commit();
  for (int k_tile = 0; k_tile < 125; ++k_tile) {
    tl::cp_async_wait<2>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)a_shared)[((k_tile % 3) * 4096)])), 1, 32, 0, 0, 4);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)b_shared)[((k_tile % 3) * 8192)])), 256, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((0 < ki) ? 1 : ((k_tile == 0) ? 0 : 1)), static_cast<uint32_t>(138478736), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(((k_tile % 6) / 3));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)a_shared)[((((((k_tile % 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((((int)blockIdx.x) + 212) / 896) * 4194304) + (((((int)blockIdx.x) + 4) & 7) * 524288)) + (i_6 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_tile * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)b_shared)[((((((((k_tile % 3) * 8192) + (((((int)threadIdx.x) & 31) >> 3) * 2048)) + (i_7 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((k_tile * 917504) + (i_7 * 114688)) + ((((int)threadIdx.x) >> 5) * 28672)) + ((((((int)blockIdx.x) + 212) % 896) >> 3) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 2752512)])));
    }
    tl::cp_async_commit();
    __syncthreads();
  }
  tl::cp_async_wait<2>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_1;
    tl::Tcgen05SMemDescriptor desc_b_1;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_1, (&(((bfloat16_t*)a_shared)[8192])), 1, 32, 0, 0, 4);
      tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)b_shared)[16384])), 256, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a_1 + (ki_1 * 32)), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, 1, static_cast<uint32_t>(138478736), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(1);
  }
  __syncthreads();
  tl::cp_async_wait<1>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_2;
    tl::Tcgen05SMemDescriptor desc_b_2;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_2, (&(((bfloat16_t*)a_shared)[0])), 1, 32, 0, 0, 4);
      tl::initialize_tcgen05_descriptor(desc_b_2, (&(((bfloat16_t*)b_shared)[0])), 256, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a_2 + (ki_2 * 32)), uint64_t(desc_b_2 + (ki_2 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, 1, static_cast<uint32_t>(138478736), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  __syncthreads();
  tl::cp_async_wait<0>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_3;
    tl::Tcgen05SMemDescriptor desc_b_3;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_3, (&(((bfloat16_t*)a_shared)[4096])), 1, 32, 0, 0, 4);
      tl::initialize_tcgen05_descriptor(desc_b_3, (&(((bfloat16_t*)b_shared)[8192])), 256, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a_3 + (ki_3 * 32)), uint64_t(desc_b_3 + (ki_3 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, 1, static_cast<uint32_t>(138478736), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  __syncthreads();
  tl::tcgen05_ld_32dp32bNx<256, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_8 = 0; i_8 < 32; ++i_8) {
    tl::store_global_256(&(*(ulonglong4*)(C + (((((((((int)blockIdx.x) + 212) / 896) * 29360128) + (((((int)blockIdx.x) + 4) & 7) * 3670016)) + (((int)threadIdx.x) * 28672)) + ((((((int)blockIdx.x) + 212) % 896) >> 3) * 256)) + (i_8 * 8)))), *(ulonglong4*)(acc + (i_8 * 8)));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 256);
  }
}

