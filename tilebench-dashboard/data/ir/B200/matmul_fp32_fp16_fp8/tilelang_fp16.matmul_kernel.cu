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

extern "C" __global__ void matmul_kernel_kernel(const half_t* __restrict__ a, const half_t* __restrict__ b, half_t* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel_kernel(const half_t* __restrict__ a, const half_t* __restrict__ b, half_t* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  float acc[512];
  half_t c_local_cast[16];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 512);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    tl::cp_async_gs<16>((&(((half_t*)a_tile)[(((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(a[((((((int)blockIdx.x) * 5242880) + (i * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)b_tile)[((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(b[((((i_1 * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)a_tile)[((((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(a[(((((((int)blockIdx.x) * 5242880) + (i_2 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)b_tile)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_3 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(b[(((((i_3 * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 318; ++k) {
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)a_tile)[((((((((k + 2) % 3) * 16384) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(a[((((((((int)blockIdx.x) * 5242880) + (i_4 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)b_tile)[(((((((((k + 2) % 3) * 16384) + (((((int)threadIdx.x) & 31) >> 3) * 4096)) + (i_5 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(b[((((((k * 262144) + (i_5 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)a_tile)[((k % 3) * 16384)])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)b_tile)[((k % 3) * 16384)])), 512, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + ((i_6 * 16384) + (ki * 32))), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_6 * 256), ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(138477584), 0, 0, 0, 0);
          }
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(((k % 6) / 3));
    }
  }
  tl::cp_async_wait<1>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_1;
    tl::Tcgen05SMemDescriptor desc_b_1;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_1, (&(((half_t*)a_tile)[0])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_1, (&(((half_t*)b_tile)[0])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a_1 + ((i_7 * 16384) + (ki_1 * 32))), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_7 * 256), 1, static_cast<uint32_t>(138477584), 0, 0, 0, 0);
        }
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  tl::cp_async_wait<0>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_2;
    tl::Tcgen05SMemDescriptor desc_b_2;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_2, (&(((half_t*)a_tile)[16384])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_2, (&(((half_t*)b_tile)[16384])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a_2 + ((i_8 * 16384) + (ki_2 * 32))), uint64_t(desc_b_2 + (ki_2 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_8 * 256), 1, static_cast<uint32_t>(138477584), 0, 0, 0, 0);
        }
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  tl::tcgen05_ld_32dp32bNx<512, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    for (int vec = 0; vec < 4; ++vec) {
      uint2 __1;
      float4 v_ = *(float4*)(acc + ((i_9 * 16) + (vec * 4)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(c_local_cast + (vec * 4)) = __1;
    }
    tl::store_global_256(&(*(ulonglong4*)(c + (((((((int)blockIdx.x) * 1048576) + ((i_9 >> 4) * 524288)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + ((i_9 & 15) * 16)))), *(ulonglong4*)(c_local_cast + 0));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 512);
  }
}

