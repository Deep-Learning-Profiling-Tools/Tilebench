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

extern "C" __global__ void bmm_kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) bmm_kernel_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_tile = ((void*)((char*)buf_dyn_shmem + 32768));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  float acc[128];
  half_t C_local_cast[16];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 128);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    tl::cp_async_gs<16>((&(((half_t*)A_tile)[(((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)B_tile)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((int)blockIdx.z) * 409600) + (i_1 * 5120)) + ((((int)threadIdx.x) >> 4) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)A_tile)[((((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_2 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)B_tile)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((((int)blockIdx.z) * 409600) + (i_3 * 5120)) + ((((int)threadIdx.x) >> 4) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 40960)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 8; ++k) {
    tl::cp_async_wait<1>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)A_tile)[((k & 1) * 8192)])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)B_tile)[((k & 1) * 8192)])), 512, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(136380432), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(((k & 3) >> 1));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)A_tile)[(((((((k & 1) * 8192) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_4 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)B_tile)[((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((((int)blockIdx.z) * 409600) + (k * 40960)) + (i_5 * 5120)) + ((((int)threadIdx.x) >> 4) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 81920)])));
    }
    tl::cp_async_commit();
    __syncthreads();
  }
  tl::cp_async_wait<1>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_1;
    tl::Tcgen05SMemDescriptor desc_b_1;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_1, (&(((half_t*)A_tile)[0])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_1, (&(((half_t*)B_tile)[0])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a_1 + (ki_1 * 32)), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, 1, static_cast<uint32_t>(136380432), 0, 0, 0, 0);
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
    tl::Tcgen05SMemDescriptor desc_a_2;
    tl::Tcgen05SMemDescriptor desc_b_2;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_2, (&(((half_t*)A_tile)[8192])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_2, (&(((half_t*)B_tile)[8192])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a_2 + (ki_2 * 32)), uint64_t(desc_b_2 + (ki_2 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, 1, static_cast<uint32_t>(136380432), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  __syncthreads();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  tl::tcgen05_ld_32dp32bNx<128, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    for (int vec = 0; vec < 4; ++vec) {
      uint2 __1;
      float4 v_ = *(float4*)(acc + ((i_6 * 16) + (vec * 4)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(C_local_cast + (vec * 4)) = __1;
    }
    tl::store_global_256(&(*(ulonglong4*)(C + (((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (((int)threadIdx.x) * 640)) + (((int)blockIdx.y) * 128)) + (i_6 * 16)))), *(ulonglong4*)(C_local_cast + 0));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 128);
  }
}

